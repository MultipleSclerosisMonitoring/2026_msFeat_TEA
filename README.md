# TFG: Framework para el Análisis Masivo de Biomarcadores Digitales de Marcha

## Descripción del Proyecto

Este proyecto implementa un **pipeline batch para el análisis de la marcha** a partir de sensores wearables (IMU y presión plantar), con el objetivo de extraer **parámetros biomecánicos clínicamente relevantes** y estimar **fatiga motora**.

El sistema está orientado a la investigación en **Esclerosis Múltiple (EM)**, permitiendo transformar grandes volúmenes de datos de sensores en **indicadores objetivos de movilidad**, siguiendo la línea metodológica de Müller et al. (2021).

---

## Objetivos

* Extracción de datos de alta frecuencia desde InfluxDB
* Procesamiento de señales inerciales y de presión
* Detección de eventos de marcha (Heel Strike, Toe Off)
* Cálculo de parámetros espaciotemporales de la marcha
* Estimación de fatiga mediante tendencias lineales (slope)
* Ejecución batch reproducible para múltiples ensayos

---

## Arquitectura del Sistema

El proyecto sigue una arquitectura modular basada en capas:

1. **Capa de Adquisición (Cloud-to-Local)**
   Extracción masiva desde InfluxDB con validación de esquemas mediante Pydantic.

2. **Capa de Persistencia (HDF5)**
   Almacenamiento eficiente de series temporales de alta frecuencia.

3. **Capa de Procesamiento (DSP Core)**
   Filtrado, calibración, segmentación y análisis de señales de marcha.

4. **Capa de Análisis Clínico**
   Cálculo de parámetros biomecánicos y métricas de fatiga.

5. **Capa de Reporte**
   Generación de métricas agregadas y visualizaciones para validación.

---

## Estructura del Proyecto

```
2026_msFeat_TEA/
├── src/gait_analysis/       # Núcleo del paquete instalable
│   ├── __init__.py
│   ├── extractor.py         # Extracción de datos desde InfluxDB
│   ├── processor.py         # Procesamiento y análisis de señales
│   └── README.md
├── scripts/                 # Herramientas CLI (ejecución batch)
│   ├── batch_extractor.py
│   ├── batch_process_all.py
│   ├── process_gait_signals.py
│   └── README.md
├── data/                    # Datos (no versionados)
├── reports/                 # Resultados y métricas
├── pyproject.toml           # Configuración del proyecto (Poetry)
├── config.yaml              # Parámetros del pipeline
├── config_db.yaml           # Credenciales de InfluxDB
└── README.md
```

---

## Instalación

### Opción 1: Instalación estándar (recomendada)

```bash
pip install .
```

### Opción 2: Entorno de desarrollo (Poetry)

```bash
poetry install
```

---

## Ejecución

Una vez instalado el paquete:

```bash
analyze-gait
```

O, en entorno de desarrollo:

```bash
poetry run analyze-gait
```

---

## Archivos de Configuración

Antes de ejecutar el pipeline, asegúrese de disponer de:

* `config_db.yaml`: credenciales de acceso a InfluxDB
* `config.yaml`: parámetros globales (frecuencia de muestreo, filtros, logging, idioma)

---

## Requisitos

* Python 3.12
* Acceso a base de datos InfluxDB

---

## Notas Técnicas

* El proyecto utiliza estructura `src` para empaquetado limpio
* La gestión de dependencias se realiza con Poetry (desarrollo)
* El sistema es instalable mediante `pip`, sin necesidad de Poetry
* El pipeline está diseñado para ejecución batch y reproducibilidad

---

## Autor

Teresa Estevan Autrán

---

## Estado del Proyecto

En desarrollo activo — implementación progresiva de algoritmos de análisis de marcha y fatiga.
