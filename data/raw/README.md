# Data Storage - Raw Signals
Este directorio gestiona el almacenamiento persistente local para los ensayos de análisis de marcha. Se utiliza un enfoque de base de datos jerárquica para manejar series temporales de alta frecuencia.

## HDF5 Database (`gait_study_data.h5`)
Este archivo actúa como el **repositorio central** y el puente crítico entre la etapa de Extracción y la etapa de Análisis Biomecánico.

### Estructura Jerárquica
Los datos se organizan siguiendo un esquema de rutas para optimizar las velocidades de acceso y búsqueda:
- `p_[SubjectID] / [TestType] / trial_[Index]`

Cada "dataset" final es un objeto **Pandas DataFrame** que incluye:
* `_time`: Estampa de tiempo sincronizada (UTC/Local).
* `S0, S1, S2...`: Señales de acelerometría y giroscopio (según la configuración del sensor).
* `lat, lng`: Datos de posicionamiento GPS (opcional).

### Ciclo de Vida de los Datos (Data Lifecycle)
1. **Source**: Poblado por `scripts/batch_extractor.py` mediante consultas Flux a InfluxDB.
2. **Consumption**: Leído por `scripts/batch_process_all.py` para la extracción de *features* clínicas.

## Integridad y Políticas
- **Formato**: Binario HDF5 comprimido (Blosc/LZ4) para minimizar el impacto en disco.
- **Política de Actualización**: Los nuevos ensayos se añaden (append) al archivo existente; las claves existentes se preservan para evitar la pérdida de datos históricos.

---
## Git & Security Note
Debido al tamaño potencial de los archivos binarios (>100MB), los archivos `.h5` están **excluidos del control de versiones** mediante el archivo `.gitignore`. 

Para regenerar la base de datos local, asegúrese de tener las credenciales correctas en `config_db.yaml` y ejecute el script de extracción masiva.

---