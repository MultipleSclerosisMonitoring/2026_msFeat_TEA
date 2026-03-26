"""
🚀 Motor de Análisis Masivo (Batch Analysis Engine) - Versión 5.1.0
------------------------------------------------------------------
Analiza todos los ensayos en HDF5 aplicando el pipeline de Müller et al. (2021).
Optimizado para rigor clínico, trazabilidad de datos y robustez de rutas.

Mejoras de rigor: Resolución de rutas híbridas, validación de contenido HDF5 
y metadatos de versión del pipeline para trazabilidad científica.

Autor: Teresa Estevan Autrán (TFG 2026)
"""

import pandas as pd
import logging
import argparse
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from gait_analysis.processor import GaitDataProcessor, ProcessConfig

PIPELINE_VERSION = "5.1.0"

def setup_logging(verbose_level):
    """Configura el sistema de logs unificado."""
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = levels.get(verbose_level, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_config(config_path):
    """Carga la configuración desde YAML con manejo de errores robusto."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
            proc_data = cfg_dict.get('processing', {})
            return ProcessConfig(**proc_data)
    except Exception as e:
        logging.warning(f"⚠️ No se pudo cargar {config_path}. Usando defaults. Error: {e}")
        return ProcessConfig()

def main():
    # 1. Configuración de Argumentos CLI
    parser = argparse.ArgumentParser(description="Gait Analysis Batch Processor")
    parser.add_argument('--config', type=str, default='config.yaml', help='Ruta al archivo de configuración (relativa o absoluta)')
    parser.add_argument('--no-plots', action='store_true', help='Desactivar gráficos para optimizar velocidad')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help='Nivel de detalle de logs')
    args = parser.parse_args()

    # 2. Inicialización y Gestión de Rutas Robusta (Punto 1: Ruta del config)
    setup_logging(args.verbose)
    logger = logging.getLogger("BatchProcessor")
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Resolución inteligente de rutas para el config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path

    h5_path = BASE_DIR / "data" / "raw" / "gait_study_data.h5"
    output_path = BASE_DIR / "reports" / "summary_metrics.csv"
    plots_dir = BASE_DIR / "reports" / "plots"
    
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # 3. Carga de Componentes
    config = load_config(config_path)
    config.verbose = args.verbose
    processor = GaitDataProcessor(config)
    results = []

    logger.info("="*60)
    logger.info(f"INICIANDO ANÁLISIS MASIVO | PIPELINE V{PIPELINE_VERSION}")
    logger.info(f"Config: {config_path}")
    logger.info("="*60)

    if not h5_path.exists():
        logger.error(f"Archivo HDF5 no detectado: {h5_path}")
        return

    try:
        with pd.HDFStore(h5_path, mode='r') as store:
            keys = store.keys()
            total = len(keys)
            
            # Punto 2: Validación de contenido del HDF5
            if total == 0:
                logger.warning("El archivo HDF5 está vacío. No hay ensayos que procesar.")
                return

            for idx, key in enumerate(keys, 1):
                try:
                    logger.info(f"[{idx}/{total}] Procesando: {key}")
                    df_raw = store.get(key)
                    
                    # Ejecución del pipeline biomecánico
                    df_proc, metrics, peaks = processor.process_signals(df_raw)
                    
                    # Auditoría Visual
                    if not args.no_plots:
                        plt.figure(figsize=(12, 4))
                        plt.plot(df_proc['_time'], df_proc['S0_filt'], label='Signal', alpha=0.7)
                        plt.plot(df_proc['_time'].iloc[peaks], df_proc['S0_filt'].iloc[peaks], "rx", label='Heel Strike')
                        plt.title(f"Trial: {key} | Peaks: {metrics.get('pasos_detectados')}")
                        plt.close() # En batch masivo, cerrar la figura es crítico para la RAM

                    # Punto 3: Trazabilidad científica
                    metrics['ensayo_id'] = key
                    metrics['pipeline_version'] = PIPELINE_VERSION
                    results.append(metrics)

                except Exception as e:
                    logger.error(f"Error procesando {key}: {str(e)}")

        # 4. Consolidación Final
        if results:
            df_final = pd.DataFrame(results)
            # Asegurar que el ID y la Versión estén al principio
            fixed_cols = ['ensayo_id', 'pipeline_version']
            other_cols = [c for c in df_final.columns if c not in fixed_cols]
            df_final[fixed_cols + other_cols].to_csv(output_path, index=False)
            
            logger.info("="*60)
            logger.info(f"REPORTE GENERADO: {output_path}")
            logger.info(f"Pipeline ejecutado sin errores fatales.")
            logger.info("="*60)
        else:
            logger.error("No se generaron resultados válidos.")

    except Exception as e:
        logger.critical(f"Fallo catastrófico en el acceso a datos: {e}")

if __name__ == "__main__":
    main()