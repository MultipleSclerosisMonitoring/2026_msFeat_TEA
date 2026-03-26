# Gait Analysis Core Package

Este paquete contiene la lógica central del pipeline de extracción y procesamiento de señales de marcha. Está diseñado como una librería independiente siguiendo estándares profesionales de Programación Orientada a Objetos (OOP) y tipado estricto.

## Componentes Principales

### 1. `extractor.py` (Motor de Adquisición)
Módulo responsable de la comunicación entre **InfluxDB** y el almacenamiento local **HDF5**.
* **Lógica de Base de Datos**: Implementa consultas Flux para recuperar datos de sensores de alta frecuencia (IMU).
* **Gestión de Persistencia**: Administra el árbol jerárquico HDF5 (`Sujeto > Test > Ensayo`) para optimizar el acceso aleatorio a grandes volúmenes de datos.
* **Sincronización Temporal**: Gestiona automáticamente la conversión entre tiempo clínico local y UTC (ISO 8601).

### 2. `processor.py` (Motor de Procesamiento de Señal)
El núcleo biomecánico encargado de transformar señales crudas en métricas clínicas exportables.
* **Filtrado Digital**: Implementa un filtro Butterworth de 4º orden de **fase cero** (`filtfilt`) para eliminar ruido sin introducir retardos temporales.
* **Autocalibración Gravitatoria**: Detecta automáticamente el eje vertical, permitiendo que el procesamiento sea independiente de la orientación física del sensor.
* **Segmentación de Marcha**: Identifica eventos de *Heel Strike* y fases de balanceo, aplicando máscaras giroscópicas para excluir giros y transiciones.



### 3. Validación y Robustez (Pydantic v2)
Para garantizar la integridad del sistema, el paquete utiliza **Pydantic** para la validación estricta de esquemas:
* **`InfluxConfig`**: Valida los archivos YAML de configuración antes de iniciar cualquier conexión de red.
* **Validación de DataFrames**: Verifica la presencia de señales críticas (S0, _time) antes de la persistencia en HDF5.

## Soporte Multi-idioma (i18n)
El paquete incluye un motor interno de internacionalización que permite conmutar los logs técnicos y las etiquetas de los reportes entre **Inglés y Español** mediante el archivo de configuración global.

## Fundamentos Científicos
Los algoritmos de este paquete están alineados con la metodología de **Müller et al. (2021)**, centrándose en la detección temprana de fatiga motora y variabilidad de la marcha en pacientes con Esclerosis Múltiple.

## Estructura Interna
```text
gait_analysis/
├── __init__.py      # Inicialización del paquete y exportación de clases
├── extractor.py     # Cliente InfluxDB y lógica de persistencia HDF5
├── processor.py     # Pipeline de DSP y extracción de métricas biomecánicas
└── README.md        # Documentación técnica del módulo (Este archivo)