"""
Herramienta CLI para Extracción Masiva de Datos Biomecánicos (Batch Extractor).

Automatiza la descarga de ensayos clínicos desde InfluxDB hacia HDF5, 
con soporte para parametrización de tests, CSV de registro, idioma y verbose.

Uso:
    python scripts/batch_extractor.py --test 6MWT --csv tests.csv --lang es --verbose 2

Autor: Teresa Estevan Autrán
Versión: 4.0.0
"""

import argparse
import urllib3
import logging
import sys
from pathlib import Path
from gait_analysis.extractor import GaitDataExtractor, get_text

# Desactivar warnings SSL (entornos corporativos/restringidos como la UPM)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def main() -> None:
    """Orquesta la extracción por lotes de datos biomecánicos."""
    
    # -----------------------------
    # 1. Definición de la interfaz CLI
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Gait Data Batch Extractor - TFG 2026",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--test", type=str, default="6MWT", help="Código del test clínico a extraer")
    parser.add_argument("--csv", type=str, default="tests.csv", help="Archivo CSV con el registro de pacientes")
    parser.add_argument("--lang", type=str, choices=['es', 'en'], default="es", help="Idioma de los logs")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1, help="Nivel de detalle de consola")

    args = parser.parse_args()

    # -----------------------------
    # 2. Inicialización del Entorno
    # -----------------------------
    # El extractor ya configura el logging internamente basado en verbose y lang
    try:
        extractor = GaitDataExtractor(
            lang=args.lang,
            verbose=args.verbose
        )
        
        # 3. Ejecución del proceso por lotes
        extractor.run_batch_extraction(
            csv_name=args.csv,
            test_type=args.test
        )
        
    except FileNotFoundError as e:
        # Errores de configuración (archivo YAML no hallado)
        logging.error(f"Error de configuración: {e}")
        sys.exit(1)
    except Exception as e:
        # Captura de cualquier otra excepción no controlada
        logging.critical(get_text('msg_error_crit', args.lang, e=str(e)))
        sys.exit(1)
    finally:
        # 4. Cierre seguro de la conexión
        if 'extractor' in locals():
            extractor.close()
            logging.info(f"--- {get_text('start_batch', args.lang, test=args.test)}: FIN ---")

if __name__ == "__main__":
    main()