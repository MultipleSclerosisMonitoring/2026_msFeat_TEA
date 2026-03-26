"""
Módulo de Extracción y Persistencia de Datos Biomecánicos (InfluxDB -> HDF5).

Este módulo implementa un motor de extracción masiva que garantiza la integridad
de las series temporales, gestiona la internacionalización de logs y centraliza
el almacenamiento en contenedores jerárquicos de alto rendimiento.

"""

import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, List, Final, Optional
from influxdb_client import InfluxDBClient
from pydantic import BaseModel, Field

# --- DICCIONARIO DE INTERNACIONALIZACIÓN (i18n) ---
TRANSLATIONS: Final = {
    'es': {
        'start_batch': "--- Iniciando extracción para test: {test} ---",
        'records_found': "Registros identificados: {n}",
        'save_ok': "[+] {id} persistido en HDF5 -> {key}",
        'no_data': "[-] {id}: El rango temporal no devolvió registros.",
        'err_query': "[!] Error en consulta Flux para {id}: {e}",
        'err_schema': "[!] {id} rechazado: Faltan señales críticas (S0/_time).",
        'err_config': "Configuración de InfluxDB no hallada: {path}"
    },
    'en': {
        'start_batch': "--- Starting batch extraction for test: {test} ---",
        'records_found': "Identified records: {n}",
        'save_ok': "[+] {id} persisted in HDF5 -> {key}",
        'no_data': "[-] {id}: Time range returned no records.",
        'err_query': "[!] Flux query error for {id}: {e}",
        'err_schema': "[!] {id} rejected: Critical signals missing (S0/_time).",
        'err_config': "InfluxDB configuration not found: {path}"
    }
}

class InfluxConfig(BaseModel):
    """Esquema de validación para parámetros de conexión a InfluxDB v2."""
    url: str
    token: str
    org: str
    bucket: str

def get_text(key: str, lang: str = 'es', **kwargs) -> str:
    """Helper para la recuperación de literales multi-idioma."""
    return TRANSLATIONS.get(lang, TRANSLATIONS['es']).get(key, key).format(**kwargs)

class GaitDataExtractor:
    """
    Motor de extracción masiva con validación de esquema y gestión de persistencia.
    """

    def __init__(
        self,
        db_config_file: str = "config/config_db.yaml",
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
        self.base_dir = Path(__file__).resolve().parents[3]
        
        # Ruta de salida HDF5 configurable
        output_h5_path = Path(output_h5)
        if not output_h5_path.is_absolute():
            output_h5_path = self.base_dir / output_h5_path
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5_database = output_h5_path

        # Ruta del YAML de conexión configurable
        config_path = Path(db_config_file)
        if not config_path.is_absolute():
            config_path = self.base_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(get_text("err_config", self.lang, path=config_path))
            
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        v_config = InfluxConfig(**raw_config["influxdb"])
        
        self._client = InfluxDBClient(
            url=v_config.url,
            token=v_config.token,
            org=v_config.org,
            timeout=60000, # Aumentado para extracciones pesadas
            verify_ssl=False
        )
        self._bucket = v_config.bucket

    def _setup_logging(self) -> logging.Logger:
        """Configura el nivel de trazabilidad basado en el parámetro verbose."""
        levels = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
        level = levels.get(self.verbose, logging.INFO)
        logging.basicConfig(level=level, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%H:%M:%S')
        return logging.getLogger(__name__)

    def _validate_extracted_data(self, df: pd.DataFrame) -> bool:
        """
        Verifica que el DataFrame contenga las señales mínimas antes de persistir.
        """
        required = {'_time', 'S0'}
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
            self.logger.error(f"Registro CSV no encontrado: {csv_path}")
            return

        df_registry = pd.read_csv(csv_path)
        subset = df_registry[df_registry["t_code"] == test_type]

        self.logger.info(get_text("start_batch", self.lang, test=test_type))
        self.logger.info(get_text("records_found", self.lang, n=len(subset)))

        for idx, row in subset.iterrows():
            self._extract_patient_data(row, idx, test_type)

    def _extract_patient_data(self, row: pd.Series, idx: int, test_type: str) -> None:
        """Consulta Flux, normalización y almacenamiento HDF5."""
        p_id = str(row['codeid'])
        
        # Gestión robusta de zonas horarias
        start_dt = pd.to_datetime(row['d_from'])
        stop_dt = pd.to_datetime(row['d_until'])
        
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize('UTC')
            stop_dt = stop_dt.tz_localize('UTC')
        
        start_iso = start_dt.isoformat()
        stop_iso = stop_dt.isoformat()

        # Query optimizada: Conservamos lat/lng para el análisis biomecánico
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
                # 1. Validación de esquema
                if not self._validate_extracted_data(df):
                    self.logger.error(get_text('err_schema', self.lang, id=p_id))
                    return

                # 2. Normalización de tiempo (Eliminamos zona horaria para compatibilidad HDF5)
                df["_time"] = df["_time"].dt.tz_localize(None)
                
                # 3. Persistencia Jerárquica
                h_key = f"p_{p_id}/{test_type}/trial_{idx}"
                df.to_hdf(self.h5_database, key=h_key, mode='a', format='table', data_columns=True)
                
                self.logger.info(get_text('save_ok', self.lang, id=p_id, key=h_key))
            else:
                self.logger.warning(get_text('no_data', self.lang, id=p_id))
                
        except Exception as e:
            self.logger.error(get_text('err_query', self.lang, id=p_id, e=str(e)))

    def close(self):
        """Finaliza la conexión con el servidor InfluxDB."""
        self._client.close()