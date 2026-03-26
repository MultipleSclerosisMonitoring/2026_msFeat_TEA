# CLI Scripts and Execution Entry Points

Este directorio contiene las herramientas de línea de comandos (CLI) diseñadas para interactuar con el paquete núcleo `gait_analysis`. Estos scripts automatizan el ciclo de vida completo de los datos: desde la recuperación en la nube hasta el almacenamiento jerárquico y el análisis biomecánico final.

## Especificaciones Técnicas
- **Lenguaje**: Python 3.12 (Estrictamente requerido).
- **Gestión de Dependencias**: Integrado con **Poetry** para una resolución de entorno determinista.
- **Motor Central**: Desacoplado, importando lógica desde el módulo `src/gait_analysis`.
- **Persistencia de Datos**: Gestión de alto rendimiento mediante HDF5 (Hierarchical Data Format).

---

## Catálogo de Scripts

### 1. `batch_extractor.py`
Herramienta profesional para la extracción masiva de señales de marcha de alta frecuencia desde InfluxDB.
- **Almacenamiento Jerárquico**: Implementa la estructura HDF5: `p_[ID_Sujeto] > [Tipo_Test] > trial_[Indice]`, optimizando las operaciones de E/S.
- **i18n & Verbose**: Soporta logs multilingües y niveles de detalle configurables vía CLI.
- **Sincronización Temporal**: Maneja automáticamente la conversión entre el tiempo clínico local y el UTC de la base de datos (ISO 8601).

### 2. `batch_process_all.py` 
El motor principal para el procesamiento por lotes de la base de datos local HDF5.
- **Autocalibración**: Detecta dinámicamente el eje vertical de cada ensayo mediante análisis gravitatorio.
- **Auditoría Visual**: Genera un gráfico de validación para cada ensayo en `reports/plots/` para verificación clínica.
- **Consolidación**: Agrega todas las características biomecánicas (Mean Stride, STD, etc.) en `reports/summary_metrics.csv`.

### 3. `process_gait_signals.py`
Herramienta de pruebas unitarias y depuración del pipeline de procesamiento.
- **Propósito**: Utilizado para ajustar filtros, frecuencias de corte y parámetros de detección de picos en un ensayo específico antes de ejecutar el análisis masivo.

---

## Guía de Ejecución

Para garantizar la correcta resolución de módulos y la carga de configuraciones, estos scripts **deben ejecutarse siempre desde el directorio raíz del proyecto** utilizando Poetry.

### 1. Extracción de Datos (Cloud a Local)
Descarga los ensayos clínicos desde InfluxDB:
```powershell
# Ejecución estándar (6MWT)
python -m poetry run python scripts/batch_extractor.py

# Ejecución personalizada (ej. Test TUG en inglés con Debug activado)
python -m poetry run python scripts/batch_extractor.py --test TUG --lang en --verbose 2