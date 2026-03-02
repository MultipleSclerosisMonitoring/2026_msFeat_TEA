# TFG: Mass Gait Signal Extraction and Feature Analysis

## Project Overview
This project focuses on the automated acquisition and processing of high-frequency gait signals from clinical tests. The system is designed to be independent, scalable, and follows professional Object-Oriented Programming (OOP) standards.

## Tech Stack
* **Language:** Python 3.12 (Strictly required)
* **Database:** InfluxDB (Flux Query Language)
* **Data Storage:** HDF5 (Hierarchical Data Format)
* **Key Libraries:** Pandas, Pydantic (v2), Influxdb-client, PyYAML, Tables.

## Project Structure
```text
2026_msFeat_TEA/
├── src/gait_analysis/       # Core package: extraction & processing logic
│   ├── extractor.py         # InfluxDB connection and HDF5 management
│   ├── processor.py         # DSP Pipeline, Auto-calibration & Feature extraction
│   └── README.md            # Technical documentation for the module
├── scripts/                 # Executable scripts (Entry points)
│   ├── batch_extractor.py   # CLI tool for massive data extraction from InfluxDB
│   ├── batch_process_all.py # Mass analysis of HDF5 trials with plot generation
│   ├── process_gait_signals.py # Unit test for single trial analysis
│   └── README.md            # Usage guide for scripts [cite: 2026-01-09]
├── data/raw/                # Local HDF5 databases (.h5)
├── reports/                 # Analysis outputs
│   ├── plots/               # Visual validation PNGs (Auto-generated)
│   └── summary_metrics.csv  # Consolidated biomechanical features
├── pyproject.toml           # Project metadata and local package config
└── README.md                # Main documentation
```

# Installation & Setup
## Create the virtual environment
python -m venv .venv

## Activate the environment (Windows PowerShell)
.\.venv\Scripts\activate

## Dependency Installation
pip install -r requirements.txt

## Local Package Installation
pip install -e .