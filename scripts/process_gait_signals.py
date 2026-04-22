"""
Pipeline de Análisis Biomecánico con Arquitectura de Configuración Externa,
Soporte Multilingüe (i18n) y Control de Verbosidad mediante Logging.

Este módulo centraliza la ejecución del procesamiento de señales de marcha,
garantizando la integridad de los datos mediante validaciones de esquema,
tipado y rango, además de proporcionar una interfaz de salida internacionalizada.

Componentes del Sistema:
    - Motor de Traducción: Gestión dinámica de literales según locale.
    - Validación de Configuración: Comprobación de tipos y dominios de valores.
    - DSP Engine: Integración con GaitDataProcessor para análisis cinemático.
    - Persistencia: Generación de artefactos científicos (PNG, CSV).
"""
"""
DEPRECATED SCRIPT

This script belongs to a legacy execution path and is no longer part of the
official pipeline.

Use the CLI entrypoints instead:
    poetry run extract-data --config config/config.yaml
    poetry run analyze-gait --config config/config.yaml
"""

import sys
import yaml
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Final

from gait_analysis.processor import GaitDataProcessor, ProcessConfig


VALID_VERBOSE_LEVELS: Final = (0, 1, 2)
DEFAULT_LANG: Final = "es"

TRANSLATIONS: Final[Dict[str, Dict[str, str]]] = {
    "es": {
        "msg_config_ok": "Configuración técnica cargada y validada correctamente.",
        "msg_reading_h5": "Accediendo a base de datos HDF5: {name}",
        "msg_proc_finish": "Análisis biomecánico finalizado: {n} pasos detectados.",
        "msg_error_crit": "Excepción crítica en el pipeline: {e}",
        "plot_title": "Segmentación de Marcha - Sujeto: {id}",
        "plot_ylabel": "Presión Normalizada (UA)",
        "plot_xlabel": "Tiempo (s)",
        "legend_hs": "Evento: Heel Strike",
        "legend_turn": "Intervalo de Giro",
    },
    "en": {
        "msg_config_ok": "Technical configuration loaded and validated successfully.",
        "msg_reading_h5": "Accessing HDF5 database: {name}",
        "msg_proc_finish": "Biomechanical analysis complete: {n} steps detected.",
        "msg_error_crit": "Critical exception in pipeline: {e}",
        "plot_title": "Gait Segmentation - Subject: {id}",
        "plot_ylabel": "Normalized Pressure (AU)",
        "plot_xlabel": "Time (s)",
        "legend_hs": "Event: Heel Strike",
        "legend_turn": "Turning Interval",
    },
}


def get_text(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANG])
    text = lang_dict.get(key, f"[{key}]")
    return text.format(**kwargs)


def setup_logging(verbose_level: int) -> logging.Logger:
    levels = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    level = levels.get(verbose_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def _validate_config_schema(config: Dict[str, Any]) -> None:
    required_top_level = ["project", "paths", "processing", "analysis"]
    for key in required_top_level:
        if key not in config:
            raise ValueError(f"Esquema incompleto: Falta sección '{key}'")

    project_cfg = config["project"]
    paths_cfg = config["paths"]
    processing_cfg = config["processing"]
    analysis_cfg = config["analysis"]

    project_schema = {
        "language": str,
        "verbose": int,
    }

    paths_schema = {
        "h5_path": str,
        "output_plot": str,
        "output_metrics": str,
    }

    processing_schema = {
        "fs": (int, float),
        "cutoff_pressure": (int, float),
        "cutoff_gyro": (int, float),
        "gyro_threshold": (int, float),
        "min_peak_distance_s": (int, float),
        "min_peak_height": (int, float),
    }

    analysis_schema = {
        "h5_key": str,
    }

    for key, expected_type in project_schema.items():
        if key not in project_cfg:
            raise ValueError(f"Falta project.{key}")
        if not isinstance(project_cfg[key], expected_type):
            raise TypeError(f"Error de tipo en project.{key}: se esperaba {expected_type}")

    for key, expected_type in paths_schema.items():
        if key not in paths_cfg:
            raise ValueError(f"Falta paths.{key}")
        if not isinstance(paths_cfg[key], expected_type):
            raise TypeError(f"Error de tipo en paths.{key}: se esperaba {expected_type}")

    for key, expected_type in processing_schema.items():
        if key not in processing_cfg:
            raise ValueError(f"Falta processing.{key}")
        if not isinstance(processing_cfg[key], expected_type):
            raise TypeError(f"Error de tipo en processing.{key}: se esperaba {expected_type}")

    for key, expected_type in analysis_schema.items():
        if key not in analysis_cfg:
            raise ValueError(f"Falta analysis.{key}")
        if not isinstance(analysis_cfg[key], expected_type):
            raise TypeError(f"Error de tipo en analysis.{key}: se esperaba {expected_type}")

    if project_cfg["verbose"] not in VALID_VERBOSE_LEVELS:
        raise ValueError(
            f"Nivel de verbosidad '{project_cfg['verbose']}' no válido. Use: {VALID_VERBOSE_LEVELS}"
        )

import warnings

warnings.warn(
    "This script is deprecated. Use the CLI entrypoints instead.",
    DeprecationWarning,
    stacklevel=2,
)

def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config.yaml"
    current_lang = DEFAULT_LANG

    try:
        if not config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no hallado: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        _validate_config_schema(config)

        project_cfg = config["project"]
        paths_cfg = config["paths"]
        processing_cfg = config["processing"]
        analysis_cfg = config["analysis"]

        current_lang = project_cfg.get("language", DEFAULT_LANG)
        logger = setup_logging(project_cfg.get("verbose", 1))

        logger.info(get_text("msg_config_ok", current_lang))

        process_params = ProcessConfig(
            fs=processing_cfg["fs"],
            cutoff_pressure=processing_cfg["cutoff_pressure"],
            cutoff_gyro=processing_cfg["cutoff_gyro"],
            gyro_threshold=processing_cfg["gyro_threshold"],
            min_peak_distance_s=processing_cfg["min_peak_distance_s"],
            min_peak_height=processing_cfg["min_peak_height"],
        )

        logger.info(
            f"Frecuencia objetivo de remuestreo: {process_params.fs:.2f} Hz | "
            f"Distancia mínima entre picos: {process_params.min_peak_distance_s:.3f} s"
        )

        processor = GaitDataProcessor(process_params)

        h5_path = base_dir / paths_cfg["h5_path"]
        h5_key = analysis_cfg["h5_key"]

        logger.debug(f"Path de acceso HDF5: {h5_path}")

        if not h5_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de entrada requerido: {h5_path}\n"
                "Genéralo primero con:\n"
                "  poetry run extract-data --config config/config.yaml\n"
                "o coloca manualmente el archivo en la ruta configurada."
            )

        logger.info(get_text("msg_reading_h5", current_lang, name=h5_path.name))
        df_raw = pd.read_hdf(h5_path, key=h5_key)

        df_proc, metrics, peaks = processor.process_signals(df_raw)
        logger.info(get_text("msg_proc_finish", current_lang, n=metrics["pasos_detectados"]))

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(df_proc["_time"], df_proc["S0_filt"], color="#1f77b4", lw=1.2)
        ax.plot(
            df_proc["_time"].iloc[peaks],
            df_proc["S0_filt"].iloc[peaks],
            "rx",
            label=get_text("legend_hs", current_lang),
            markersize=8,
        )

        ax.fill_between(
            df_proc["_time"],
            0,
            df_proc["S0_filt"].max(),
            where=df_proc["is_turning"],
            color="red",
            alpha=0.1,
            label=get_text("legend_turn", current_lang),
        )

        ax.set_title(get_text("plot_title", current_lang, id=h5_key))
        ax.set_ylabel(get_text("plot_ylabel", current_lang))
        ax.set_xlabel(get_text("plot_xlabel", current_lang))
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

        plot_out = base_dir / paths_cfg["output_plot"]
        csv_out = base_dir / paths_cfg["output_metrics"]

        plot_out.parent.mkdir(parents=True, exist_ok=True)
        csv_out.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(plot_out, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame([metrics]).to_csv(csv_out, index=False)
        logger.debug(f"Persistencia completada en directorio: {plot_out.parent}")

    except Exception as e:
        logging.error(get_text("msg_error_crit", current_lang, e=str(e)), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()