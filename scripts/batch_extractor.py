# Este archivo hace de batch extractor para extraer datos de multiples archivos de influxDB
# Automatiza la extraccion de series temporales de alta frecuencia desde influxDB

# batch_extractor.py - Módulo de extracción masiva para el TFG 2026
# batch_extractor.py - Módulo de extracción masiva parametrizado para el TFG 2026
import argparse
import sys
import urllib3
from pathlib import Path

# Configuración del entorno para encontrar el paquete en /src
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / "src"))

from gait_analysis.extractor import GaitDataExtractor


# Silenciar avisos de conexiones HTTPS no verificadas para el entorno de la UPM
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Gestión de parámetros desde terminal (Command Line Interface)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Herramienta de extracción masiva TFG.")
    
    # Estos son los parámetros que el profesor podrá configurar
    parser.add_argument("--test", type=str, default="6MWT", help="Tipo de test (6MWT, TUG, etc.)")
    parser.add_argument("--csv", type=str, default="tests.csv", help="Nombre del archivo de registro CSV.")
    parser.add_argument("--out", type=str, default="data/raw", help="Carpeta de destino para los archivos.")
    
    args = parser.parse_args()

    # Creación y ejecución usando los parámetros introducidos
    extractor = GaitDataExtractor(output_folder=args.out)
    extractor.run_batch_extraction(csv_path=args.csv, test_type=args.test)
    extractor.close()
    print("\nProceso finalizado.")