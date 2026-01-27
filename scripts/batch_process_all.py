"""
Script de Procesamiento Masivo de Marcha (Batch Processing)
----------------------------------------------------------
Este script automatiza el análisis de todos los ensayos contenidos en un 
archivo HDF5. Recorre cada 'key', extrae las métricas biomecánicas y 
consolida los resultados en un único reporte CSV para análisis estadístico.
"""
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from gait_analysis.processor import GaitDataProcessor, ProcessConfig

# Configurar logging para ver el progreso
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Ejecuta el pipeline de análisis sobre la base de datos completa.
    
    Flujo de trabajo:
        1. Localiza el archivo HDF5 en la estructura de carpetas.
        2. Inicializa el procesador con la configuración por defecto.
        3. Itera sobre cada ensayo (key) dentro del almacén HDF5.
        4. Gestiona errores de forma individual para no interrumpir el lote.
        5. Exporta un DataFrame consolidado a un archivo CSV.
    """
    # 1. Gestión de rutas relativas mediante Pathlib
    BASE_DIR = Path(__file__).resolve().parent.parent
    h5_path = BASE_DIR / "data" / "raw" / "gait_study_data.h5"
    output_path = BASE_DIR / "reports" / "summary_metrics.csv"
    plots_dir = BASE_DIR / "reports" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 2. Inicialización del procesador profesional
    processor = GaitDataProcessor(ProcessConfig())
    results = []

    print(f" Iniciando procesamiento por lotes: {h5_path.name}")

    try:
        # Abrir el almacén de datos en modo solo lectura para evitar bloqueos
        with pd.HDFStore(h5_path, mode='r') as store:
            keys = store.keys()
            print(f"Se han detectado {len(keys)} ensayos para procesar.\n")
            for key in keys:
                try:
                    logger.info(f"Analizando ensayo: {key}")
                    df = store.get(key)
                    df_proc, metrics, peaks = processor.process_signals(df)
                    
                    plt.figure(figsize=(12, 4))
                    plt.plot(df_proc['_time'], df_proc['S0_filt'], label='Señal Presión')
                    plt.plot(df_proc['_time'].iloc[peaks], df_proc['S0_filt'].iloc[peaks], "rx", label='Pasos')
                    
                    safe_name = key.replace('/', '_').strip('_')
                    plt.title(f"Ensayo: {key} | Pasos: {metrics['pasos_detectados']}")
                    plt.savefig(plots_dir / f"{safe_name}.png")
                    plt.close()

                    metrics['ensayo_id'] = key
                    results.append(metrics)
            
                except Exception as e:
                    # Captura errores específicos sin detener el procesamiento del resto
                    logger.error(f"Error crítico en ensayo {key}: {e}")

    # 4. Consolidación y exportación de resultados
        if results:
            df_results = pd.DataFrame(results)
            
            # Reordenación de columnas para priorizar la ID del ensayo
            cols_order = ['ensayo_id'] + [c for c in df_results.columns if c != 'ensayo_id']
            df_results = df_results[cols_order]
            
            # Persistencia en CSV
            df_results.to_csv(output_path, index=False)
            
            print(f"\n PROCESO COMPLETADO CON ÉXITO")
            print(f" Informe consolidado guardado en: {output_path}")
            print("-" * 50)
            print(df_results.head()) # Muestra las primeras filas como vista previa

    except FileNotFoundError:
        print(f"Error fatal: No se encontró el archivo HDF5 en {h5_path}")
    except Exception as e:
        print(f"Error inesperado durante la ejecución: {e}")

if __name__ == "__main__":
    main()