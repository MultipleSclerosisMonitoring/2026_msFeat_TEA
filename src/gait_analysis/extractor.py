"""
Módulo de Extracción y Persistencia de Datos Biomecánicos (InfluxDB -> HDF5).

Este módulo implementa un motor de extracción masiva que garantiza la integridad
de las series temporales, gestiona la internacionalización de logs y centraliza
el almacenamiento en contenedores jerárquicos de alto rendimiento.
"""

import logging
from pathlib import Path

import pandas as pd
from influxdb_client import InfluxDBClient
from pydantic import BaseModel

from gait_analysis.utils.messages import get_message


class InfluxConfig(BaseModel):
    """Esquema de validación para parámetros de conexión a InfluxDB v2."""

    url: str
    token: str
    org: str
    bucket: str


class GaitDataExtractor:
    """
    Motor de extracción masiva con validación de esquema y gestión de persistencia.
    """

    def __init__(
        self,
        influx_config: dict,
        output_h5: str | Path = "data/raw/gait_study_data.h5",
        lang: str = "es",
        verbose: int = 1,
    ):
        """
        Inicializa el cliente de InfluxDB y valida el entorno de salida.
        """
        self.lang = lang
        self.verbose = verbose
        self.logger = self._setup_logging()

        # Resolución de rutas mediante Pathlib
        self.base_dir = Path(__file__).resolve().parents[2]

        # Ruta de salida HDF5 configurable
        output_h5_path = Path(output_h5)
        if not output_h5_path.is_absolute():
            output_h5_path = self.base_dir / output_h5_path
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5_database = output_h5_path

        try:
            v_config = InfluxConfig(**influx_config)
        except Exception as e:
            raise ValueError(f"Invalid InfluxDB configuration: {e}") from e

        self._client = InfluxDBClient(
            url=v_config.url,
            token=v_config.token,
            org=v_config.org,
            timeout=60000,
            verify_ssl=False,
        )
        self._bucket = v_config.bucket

    def _setup_logging(self) -> logging.Logger:
        """Configura el nivel de trazabilidad basado en el parámetro verbose."""
        levels = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG}
        level = levels.get(self.verbose, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%H:%M:%S",
        )
        return logging.getLogger(__name__)

    def _validate_extracted_data(self, df: pd.DataFrame) -> bool:
        """
        Verifica que el DataFrame contenga las señales mínimas antes de persistir.
        """
        required = {"_time", "S0"}
        return required.issubset(df.columns)

    def run_batch_extraction(
        self,
        csv_filepath: str | Path = "tests.csv",
        test_type: str = "6MWT",
    ) -> None:
        """Ejecuta el pipeline de extracción masiva definido en el registro CSV."""
        csv_path = Path(csv_filepath)
        if not csv_path.is_absolute():
            csv_path = self.base_dir / csv_path

        if not csv_path.exists():
            raise FileNotFoundError(
                get_message("csv_not_found", self.lang, path=csv_path)
            )

        df_registry = pd.read_csv(csv_path)
        subset = df_registry[df_registry["t_code"] == test_type]

        if subset.empty:
            raise ValueError(
                f"No records found in registry CSV for test_type='{test_type}'."
            )

        self.logger.info(get_message("start_batch", self.lang, test=test_type))
        self.logger.info(get_message("records_found", self.lang, n=len(subset)))

        saved_count = 0

        for idx, row in subset.iterrows():
            if self._extract_patient_data(row, idx, test_type):
                saved_count += 1

        if saved_count == 0:
            raise RuntimeError(
                f"Batch extraction finished but no trials were stored in {self.h5_database}."
            )

        self.logger.info(f"Stored {saved_count} trials in {self.h5_database}")

    def _extract_patient_data(self, row: pd.Series, idx: int, test_type: str) -> bool:
        """Consulta Flux, normalización y almacenamiento HDF5."""
        p_id = str(row["codeid"])

        start_dt = pd.to_datetime(row["d_from"])
        stop_dt = pd.to_datetime(row["d_until"])

        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize("UTC")
            stop_dt = stop_dt.tz_localize("UTC")

        start_iso = start_dt.isoformat()
        stop_iso = stop_dt.isoformat()

        query = f'''
        from(bucket: "{self._bucket}")
          |> range(start: {start_iso}, stop: {stop_iso})
          |> filter(fn: (r) => r["_measurement"] == "Gait")
          |> filter(fn: (r) => r["CodeID"] == "{p_id}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> drop(columns: ["_start", "_stop", "_measurement", "result", "table", "CodeID", "app", "mac"])
        '''

        try:
            df = self._client.query_api().query_data_frame(query)

            if isinstance(df, pd.DataFrame) and not df.empty:
                if not self._validate_extracted_data(df):
                    self.logger.error(get_message("err_schema", self.lang, id=p_id))
                    return False

                df["_time"] = df["_time"].dt.tz_localize(None)

                h_key = f"p_{p_id}/{test_type}/trial_{idx}"
                df.to_hdf(
                    self.h5_database,
                    key=h_key,
                    mode="a",
                    format="table",
                    data_columns=True,
                )

                self.logger.info(get_message("save_ok", self.lang, id=p_id, h5_key=h_key))
                return True

            self.logger.warning(get_message("no_data", self.lang, id=p_id))
            return False

        except Exception as e:
            self.logger.error(get_message("err_query", self.lang, id=p_id, e=str(e)))
            return False

    def close(self) -> None:
        """Finaliza la conexión con el servidor InfluxDB."""
        self._client.close()