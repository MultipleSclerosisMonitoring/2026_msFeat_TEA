# TFG: Framework para el Análisis Masivo de Biomarcadores Digitales de Marcha

## Descripción del Proyecto
Este ecosistema de software ha sido desarrollado para automatizar el ciclo de vida completo de los datos biomecánicos: desde la ingesta de telemetría de alta frecuencia en la nube hasta la generación de métricas clínicas validadas. 

El sistema está diseñado específicamente para la investigación en **Esclerosis Múltiple (EM)**, permitiendo procesar grandes volúmenes de ensayos clínicos y transformarlos en indicadores objetivos de movilidad y fatiga motora.



---

## Arquitectura del Sistema

El proyecto sigue una arquitectura de **capas desacopladas** para garantizar la mantenibilidad y escalabilidad del software:

1.  **Capa de Adquisición (Cloud-to-Local):** Extracción masiva desde InfluxDB con validación de esquemas vía Pydantic v2.
2.  **Capa de Persistencia (HDF5):** Almacenamiento jerárquico optimizado para series temporales masivas, evitando la sobrecarga de formatos planos.
3.  **Capa de Procesamiento (DSP Core):** Algoritmos de filtrado, autocalibración gravitatoria y segmentación de eventos de marcha.
4.  **Capa de Reporte:** Generación automatizada de reportes CSV y auditorías visuales (Plots) para validación clínica.

### Estructura de Directorios
```text
2026_msFeat_TEA/
├── src/gait_analysis/       # CORE: Paquete lógico instalable
│   ├── __init__.py          # Inicializador y exportación de clases
│   ├── extractor.py         # Cliente InfluxDB y gestión de HDF5
│   ├── processor.py         # Algoritmos biomecánicos y DSP
│   └── README.md            # Documentación técnica del módulo central
├── scripts/                 # CLI: Herramientas de ejecución
│   ├── batch_extractor.py   # Descarga masiva de ensayos (Influx -> HDF5)
│   ├── batch_process_all.py # Motor de análisis masivo (v5.1.0)
│   ├── process_gait_signals.py # Herramienta de test y depuración unitaria
│   └── README.md            # Guía de uso para ejecución de scripts
├── data/                    # STORAGE: (Ignorado por Git)
│   └── raw/                 # Bases de datos jerárquicas (.h5)
├── reports/                 # OUTPUTS: Resultados del estudio
│   ├── plots/               # Gráficos de validación de pasos (Heel Strikes)
│   └── summary_metrics.csv  # Reporte maestro final consolidado
├── pyproject.toml           # Configuración de Poetry (Python 3.12)
├── config.yaml              # Parámetros globales (i18n, filtros, etc.)
├── config_db.yaml           # Credenciales de base de datos
└── README.md                # Documentación principal (Este archivo)
```

# Instalacion y Despliegue
Este proyecto usa **Poetry** para manejar dependencias y entornos virtuales, asegurando reproducibilidad total.

### 1. Configuración del Entorno
Desde la raíz del proyecto (`2026_msFeat_TEA/`), ejecute el siguiente comando para crear el entorno virtual e instalar todas las dependencias:

```powershell
# Este comando crea el entorno virtual e instala todas las dependencias del proyecto
python -m poetry install
```

### 2. Archivos de Configuración
Asegúrese de contar con los siguientes archivos en el directorio raíz antes de iniciar:

config_db.yaml: Credenciales de acceso a InfluxDB (URL, Token, Organización y Bucket).

config.yaml: Parámetros globales del pipeline (Frecuencia de muestreo, frecuencias de corte de los filtros, idioma de los reportes y nivel de detalle del logging).

### 3. Guía de Ejecución
Para garantizar que se utilice el entorno virtual correcto y el paquete gait_analysis sea reconocido, ejecute siempre los scripts mediante poetry run:
```powershell
python -m poetry run python scripts/batch_extractor.py
```