# Execution Scripts & Entry Points

This directory contains the executable scripts designed to interface with the `gait_analysis` core package. These tools automate the data lifecycle, from cloud retrieval to local hierarchical storage.

## Technical Specifications
- **Language**: Python 3.12
- **Core Engine**: Integrated with `src.gait_analysis` module.
- **Validation**: Pydantic v2 (Strict schema validation for database connectivity).
- **Persistence**: HDF5 (Hierarchical Data Format) for high-performance biomechanical data management.

## Script Catalog

### 1. `batch_extractor.py`
A professional-grade CLI tool for mass extraction of high-frequency gait signals from InfluxDB.
- **Hierarchical Storage**: Implements the HDF5 structure: `p_[SubjectID] > [TestType] > trial_[Index]`. This replaces legacy Excel formats to optimize I/O operations.
- **Robust Validation**: Leverages the `InfluxConfig` Pydantic model to validate YAML credentials before initiating network requests.
- **Environment Sync**: Handles automatic conversion between Clinical Local Time (CET/CEST) and Database UTC (ISO 8601).
- **CLI Parameterization**: Fully configurable via `argparse` for flexible batch processing.

<<<<<<< HEAD
### 2. `process_gait_signals.py` (In Development)
Module dedicated to digital signal processing, including:
- Butterworth filtering for noise reduction.
- Gyroscope-based turn detection.
- Gait event identification (Heel Strike / Toe Off).
=======
### 2. `batch_process_all.py` 
The main analysis engine for mass processing of the local HDF5 database.
- **Workflow**: Iterates through all trials, applies the DSP pipeline, and exports results.
- **Auto-Calibration**: Automatically detects the vertical axis (Ax, Ay, or Az) for each trial.
- **Visual Audit**: Generates a validation plot for every trial in `reports/plots/`.
- **Data Consolidation**: Aggregates all biomechanical features into `reports/summary_metrics.csv`.

### 3. `process_gait_signals.py`
Unit testing and debugging tool for the processing pipeline.
- **Purpose**: Used to fine-tune filters and peak detection parameters on a single specific trial before running a full batch.
>>>>>>> origin/master

## Usage
To ensure correct module resolution, these scripts must be executed from the **project root** directory.

### 1. Environment Setup
Activate your virtual environment and ensure the local package is installed:
```powershell
# Activate environment (Windows)
.\.venv\Scripts\activate

# Install the project in editable mode (CRUCIAL for scripts to work)
pip install -e . 
```

### 2. Running Extraction
You can run the extractor using default values or custom parameters:
```powershell
# Run with default settings (6MWT)
python scripts/batch_extractor.py

# Custom execution for specific tests
python scripts/batch_extractor.py --test TUG --csv tests.csv --out data/raw

# View all available options
python scripts/batch_extractor.py --help
<<<<<<< HEAD
```
=======
```

### 2. Running Mass Processing & Analysis
```powershell
    # Run the complete analysis batch
    python scripts/batch_process_all.py
```
>>>>>>> origin/master
