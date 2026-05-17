# Scripts

## Overview

This directory contains auxiliary scripts for batch processing and
diagnostic analysis. They complement the main CLI commands but are
not part of the core installable package.

## run_batch.py

Processes all Left-foot HDF5 keys in the dataset and consolidates
results into a single CSV file.

```bash
python scripts/run_batch.py --config config/config.yaml
```

Optional flags:

- `--no-plots` — skip plot generation (faster for large datasets)
- `--test-type 6MWT` — filter by clinical test type
- `--output path/to/output.csv` — override output path

Output:
- `reports/data/batch_metrics.csv` — one row per trial with all metrics
- `reports/data/per_minute_<key>.csv` — per-block fatigue metrics per trial
- `reports/plots/batch/<key>.png` — segmentation plots (if --no-plots not set)

## diagnostics/

Standalone scripts for model calibration and signal inspection.
Run from the project root with `PYTHONPATH=src`.

| Script | Purpose |
|---|---|
| `calibrate_spatial_models.py` | Calibrates K constants for gyro-norm and biometric spatial models using GPS reference trials. Generates `reports/plots/spatial_model_calibration.png`. |
| `inspect_madgwick.py` | Diagnoses Madgwick filter behaviour and accelerometer saturation. Generates `reports/plots/madgwick_diagnosis.png`. |
| `inspect_swing_angle.py` | Inspects gyroscope-based swing angle estimation for the pendulum model. |
| `validate_spatial.py` | Validates spatial metrics (walking speed, stride length) against known-distance reference trials (TUG, T25FW). |

### Example usage

```bash
# Windows PowerShell
$env:PYTHONPATH="src"
python scripts/diagnostics/calibrate_spatial_models.py
python scripts/diagnostics/inspect_madgwick.py
```