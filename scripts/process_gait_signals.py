"""
Script de Ejecución: Procesamiento de Marcha Automatizado
---------------------------------------------------------
Este script carga datos jerárquicos desde HDF5, aplica el pipeline de análisis
con autocalibración de ejes y genera reportes visuales persistentes.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gait_analysis.processor import GaitDataProcessor, ProcessConfig

def main() -> None:
    """Punto de entrada principal para el análisis de ensayos.
    
    Flujo:
        1. Resolución de rutas mediante pathlib (multiplataforma).
        2. Configuración del procesador.
        3. Ejecución del análisis y persistencia de resultados en disco.
    """
    # 1. Gestión de rutas relativas
    BASE_DIR = Path(__file__).resolve().parent.parent
    h5_path = BASE_DIR / "data" / "raw" / "gait_study_data.h5"
    
    output_dir = BASE_DIR / "reports" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Inicialización profesional del procesador
    config = ProcessConfig(save_plots=True)
    processor = GaitDataProcessor(config)

    try:
        # 3. Carga de datos desde base de datos HDF5
        print(f"Cargando ensayo desde: {h5_path}")
        test_key = 'p_SOMHUG003-31/6MWT/trial_21'
        df = pd.read_hdf(h5_path, key=test_key, mode='r')

        # 4. Procesamiento inteligente (Autocalibrado)
        df_proc, metrics, peaks = processor.process_signals(df)

        # 5. Generación de reporte visual (Modo Headless)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Gráfico de Presión
        ax1.plot(df_proc['_time'], df_proc['S0_filt'], label='Presión Filtrada (S0)', color='navy')
        ax1.plot(df_proc['_time'].iloc[peaks], df_proc['S0_filt'].iloc[peaks], "rx", label='Heel Strike')
        ax1.fill_between(df_proc['_time'], 0, df_proc['S0_filt'].max(), 
                         where=df_proc['is_turning'], color='red', alpha=0.15, label='Giro (Excluido)')
        ax1.set_title(f"Detección de Pasos: {metrics['pasos_detectados']} eventos")
        ax1.legend()

        # Gráfico de Giroscopio
        ax2.plot(df_proc['_time'], df_proc['Gz_filt'], label='Giroscopio Z', color='darkorange')
        ax2.set_title(f"Análisis de Estabilidad | Eje Vertical: {metrics['eje_vertical_utilizado']}")
        ax2.legend()

        # 6. Persistencia de resultados
        file_name = output_dir / "analisis_final_marcha.png"
        plt.tight_layout()
        plt.savefig(file_name, dpi=300)
        plt.close()
        
        print(f"Proceso completado exitosamente.")
        print(f"Métricas: {metrics}")
        print(f"Gráfico guardado en: {file_name}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo HDF5 en {h5_path}")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()