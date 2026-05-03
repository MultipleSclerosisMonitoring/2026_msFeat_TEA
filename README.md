# MS-Feat: A Modular Framework for Gait Analysis

## Overview

MS-Feat is a modular batch-processing framework for gait analysis based on wearable sensors (IMU and plantar pressure). The system is designed to extract clinically relevant biomechanical parameters and estimate motor fatigue from walking tests.

The framework is oriented towards research in neurological conditions such as Multiple Sclerosis (MS), enabling the transformation of large-scale sensor data into objective mobility indicators.

---

## Objectives

- High-frequency data extraction from InfluxDB
- Processing of inertial and plantar pressure signals
- Detection of Heel Strike events from plantar pressure
- Computation of spatiotemporal gait parameters
- Baseline estimation of fatigue-sensitive temporal trends
- Reproducible batch execution across multiple trials

---

## System Architecture

The project follows a modular layered architecture:

1. **Acquisition Layer (Cloud-to-Local)**  
   Data extraction from InfluxDB.

2. **Persistence Layer (HDF5)**  
   Efficient storage of time-series data.

3. **Processing Layer (DSP Core)**  
   Signal filtering, calibration, segmentation, and gait analysis.

4. **Clinical Analysis Layer**  
   Computation of biomechanical parameters and fatigue metrics.

5. **Reporting Layer**  
   Generation of aggregated metrics and validation outputs.

Detailed architecture and diagrams are available in the Sphinx documentation:
`docs/_build/html/index.html`

## Project Structure
```
2026_msFeat_TEA/
├── src/gait_analysis/ # Core installable package
│ ├── __init__.py 
│ ├── extractor.py # Data extraction from InfluxDB
│ ├── processor.py # Signal processing and analysis
│ ├── cli/ # Command-line interfaces
│ │ ├── extract_data.py
│ │ └── analyze_gait.py
│ └── README.md
├── config/
│ ├── config.example.yaml # Versioned template configuration
│ └── config.yaml # Local configuration (not versioned)
├── data/ # Input data (ignored by Git)
├── reports/ # Output results and metrics
├── docs/ # Sphinx documentation
├── pyproject.toml # Project configuration (Poetry)
└── README.md
```

## Installation

### Recommended (Poetry)

```bash
poetry install
```
### Alternative (standard)
```bash
python -m pip install .
```

## Configuration 
The pipeline is fully driven by external configuration.
### 1. Create your configuration file
```bash
cp config/config.example.yaml config/config.yaml
cp .env.example .env
```
### 2. Edit the configuration file
Define:
- InfluxDB credentials (via `.env`)
- Input/output paths
- Signal processing parameters
- Logging verbosity
- Language settings
### 3. Fill in the secrets in `.env`
The CLI automatically loads a project-local `.env` file and resolves
`${VAR_NAME}` placeholders from `config/config.yaml`.

This keeps credentials out of versioned YAML files while preserving a
reproducible config structure.
#### Important:
The file `.env` contains sensitive information and must not be committed to version control.

## Usage 
All commands must be executed within the Poetry environment.

### Data extraction (InfluxDB → HDF5)
```bash
poetry run extract-data --config config/config.yaml
```

Selective extraction and HDF5 audit are also supported:

```bash
poetry run extract-data --config config/config.yaml --ids 87 89
poetry run extract-data --config config/config.yaml --codeid RHRHUG004-1 --test 6MWT
poetry run extract-data --config config/config.yaml --ids 87 89 --check-only
poetry run extract-data --config config/config.yaml --test 6MWT --missing-only
poetry run extract-data --config config/config.yaml --list-hdf5-keys
```

### Typical targeted workflow
For day-to-day work, the recommended flow is usually:

1. Check whether the selected CSV rows are already present in HDF5:

```bash
poetry run extract-data --config config/config.yaml --ids 87 89 --check-only
```

2. Extract only the missing or desired cases:

```bash
poetry run extract-data --config config/config.yaml --ids 87 89
```

3. Inspect the available HDF5 keys if needed:

```bash
poetry run extract-data --config config/config.yaml --list-hdf5-keys
```

4. Analyze a specific extracted foot directly from the command line:

```bash
poetry run analyze-gait --config config/config.yaml \
  --h5-key p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Right
```

### Gait analysis and processing
```bash
poetry run analyze-gait --config config/config.yaml
```

You can analyze a specific extracted trial without editing the config file:

```bash
poetry run analyze-gait --config config/config.yaml \
  --h5-key p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Right
```
### Important note
The file data/raw/gait_study_data.h5 is not included in the repository.
If it does not exist, the analysis command will fail with an explicit error indicating how to regenerate it:
```bash
poetry run extract-data --config config/config.yaml
```
This ensures that the pipeline can always be executed from scratch.

## Documentation (Sphinx)

The documentation is built using Sphinx.

### Build locally

```bash
python -m sphinx -b html docs/source docs/_build/html
```

## Requirements
- Python 3.12
- Access to an InfluxDB instance

## Technical Notes
- The project follows a src-based layout for clean packaging
- Dependency management is handled via Poetry
- The system can also be installed using pip
- All runtime parameters are externalized in a single YAML configuration file
- The pipeline is designed for batch execution and reproducibility
- The CLI layer ensures consistent execution across environments

## Author
Teresa Estevan Autrán
