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

Autor: Teresa Estevan Autrán
Versión: 5.0.0
"""

import yaml
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Final
from gait_analysis.processor import GaitDataProcessor, ProcessConfig

# --- CONSTANTES Y DICCIONARIO DE INTERNACIONALIZACIÓN (i18n) ---

VALID_VERBOSE_LEVELS: Final = (0, 1, 2)
DEFAULT_LANG: Final = 'es'

TRANSLATIONS: Final[Dict[str, Dict[str, str]]] = {
    'es': {
        'msg_config_ok': "Configuración técnica cargada y validada correctamente.",
        'msg_reading_h5': "Accediendo a base de datos HDF5: {name}",
        'msg_proc_finish': "Análisis biomecánico finalizado: {n} pasos detectados.",
        'msg_error_crit': "Excepción crítica en el pipeline: {e}",
        'plot_title': "Segmentación de Marcha - Sujeto: {id}",
        'plot_ylabel': "Presión Normalizada (UA)",
        'plot_xlabel': "Tiempo (s)",
        'legend_hs': "Evento: Heel Strike",
        'legend_turn': "Intervalo de Giro"
    },
    'en': {
        'msg_config_ok': "Technical configuration loaded and validated successfully.",
        'msg_reading_h5': "Accessing HDF5 database: {name}",
        'msg_proc_finish': "Biomechanical analysis complete: {n} steps detected.",
        'msg_error_crit': "Critical exception in pipeline: {e}",
        'plot_title': "Gait Segmentation - Subject: {id}",
        'plot_ylabel': "Normalized Pressure (AU)",
        'plot_xlabel': "Time (s)",
        'legend_hs': "Event: Heel Strike",
        'legend_turn': "Turning Interval"
    }
}

def get_text(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    """
    Recupera el literal correspondiente al idioma configurado con fallback seguro.

    Args:
        key (str): Identificador del mensaje.
        lang (str): Código de idioma configurado.
        **kwargs: Parámetros para interpolación de strings.

    Returns:
        str: Texto traducido o clave de emergencia entre corchetes si no existe.
    """
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANG])
    text = lang_dict.get(key, f"[{key}]")
    return text.format(**kwargs)

# --- CONFIGURACIÓN DE SISTEMA Y VALIDACIÓN ---

def setup_logging(verbose_level: int) -> logging.Logger:
    """
    Inicializa el sistema de logging basándose en el nivel de verbosidad validado.

    Args:
        verbose_level (int): Nivel de detalle (0: ERROR, 1: INFO, 2: DEBUG).

    Returns:
        logging.Logger: Instancia configurada para la trazabilidad del módulo.
    """
    levels = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    level = levels.get(verbose_level, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)

def _validate_config_schema(config: Dict[str, Any]) -> None:
    """
    Valida la integridad estructural, tipos y dominios del archivo de configuración.

    Args:
        config (Dict[str, Any]): Estructura de datos cargada desde YAML.

    Raises:
        ValueError: Si faltan claves o los valores están fuera de dominio.
        TypeError: Si los tipos de datos no coinciden con el esquema esperado.
    """
    required_schema = {
        'fs': (int, float),
        'cutoff_pressure': (int, float),
        'cutoff_gyro': (int, float),
        'gyro_threshold': (int, float),
        'h5_path': str,
        'h5_key': str,
        'output_plot': str,
        'output_metrics': str,
        'language': str,
        'verbose': int
    }
    
    for key, expected_type in required_schema.items():
        if key not in config:
            raise ValueError(f"Esquema incompleto: Falta parámetro '{key}'")
        if not isinstance(config[key], expected_type):
            raise TypeError(f"Error de tipo en '{key}': Se esperaba {expected_type}")
    
    if config['verbose'] not in VALID_VERBOSE_LEVELS:
        raise ValueError(f"Nivel de verbosidad '{config['verbose']}' no válido. Use: {VALID_VERBOSE_LEVELS}")

# --- PUNTO DE ENTRADA PRINCIPAL ---

def main() -> None:
    """
    Orquesta el flujo de trabajo del pipeline biomecánico.
    
    Gestiona el ciclo de vida de los datos desde la ingesta HDF5 hasta la 
    generación de reportes, aplicando políticas de i18n y manejo robusto de errores.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "config.yaml"
    
    # Inicialización de idioma por defecto para errores tempranos
    current_lang = DEFAULT_LANG

    try:
        # 1. Carga de Configuración y Configuración de Sistema
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Archivo de configuración no hallado: {CONFIG_PATH}")
            
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        _validate_config_schema(config)
        
        current_lang = config.get('language', DEFAULT_LANG)
        logger = setup_logging(config.get('verbose', 1))
        
        logger.info(get_text('msg_config_ok', current_lang))

        # 2. Configuración e Ingesta de Datos
        process_params = ProcessConfig(
            fs=config['fs'],
            cutoff_pressure=config['cutoff_pressure'],
            cutoff_gyro=config['cutoff_gyro'],
            gyro_threshold=config['gyro_threshold']
        )
        processor = GaitDataProcessor(process_params)

        h5_path = BASE_DIR / config['h5_path']
        logger.debug(f"Path de acceso HDF5: {h5_path}")
        logger.info(get_text('msg_reading_h5', current_lang, name=h5_path.name))
        
        df_raw = pd.read_hdf(h5_path, key=config['h5_key'])
        
        # 3. Procesamiento y Extracción de Métricas
        df_proc, metrics, peaks = processor.process_signals(df_raw)
        logger.info(get_text('msg_proc_finish', current_lang, n=metrics['pasos_detectados']))

        # 4. Generación y Persistencia de Reportes
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(df_proc['_time'], df_proc['S0_filt'], color='#1f77b4', lw=1.2)
        ax.plot(df_proc['_time'].iloc[peaks], df_proc['S0_filt'].iloc[peaks], "rx", 
                label=get_text('legend_hs', current_lang), markersize=8)
        
        ax.fill_between(df_proc['_time'], 0, df_proc['S0_filt'].max(), 
                         where=df_proc['is_turning'], color='red', alpha=0.1, 
                         label=get_text('legend_turn', current_lang))
        
        ax.set_title(get_text('plot_title', current_lang, id=config['h5_key']))
        ax.set_ylabel(get_text('plot_ylabel', current_lang))
        ax.set_xlabel(get_text('plot_xlabel', current_lang))
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)

        # 5. Exportación de Artefactos
        plot_out = BASE_DIR / config['output_plot']
        csv_out = BASE_DIR / config['output_metrics']
        
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        csv_out.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(plot_out, dpi=300, bbox_inches='tight')
        plt.close(fig)

        pd.DataFrame([metrics]).to_csv(csv_out, index=False)
        logger.debug(f"Persistencia completada en directorio: {plot_out.parent}")

    except Exception as e:
        # Uso de exc_info=True para capturar traceback completo en modo DEBUG
        logging.error(get_text('msg_error_crit', current_lang, e=str(e)), exc_info=True)

if __name__ == "__main__":
    main()