# Data Storage - Raw Signals

This directory contains the raw input data consumed by the MS-Feat pipeline.
Its contents are excluded from version control via `.gitignore` due to size
and sensitivity.

## HDF5 Database (`gait_study_data.h5`)

Central storage layer between the extraction stage (InfluxDB) and the
biomechanical analysis stage. This file is the canonical input of the
`analyze-gait` CLI.

### Hierarchical Structure

Data is organized using a hierarchical key scheme optimized for fast access:

`p_[SubjectID] / [TestType] / trial_[Index]`

Each dataset is a Pandas DataFrame with the following columns:

- `_time`: timestamp (UTC, timezone-stripped for HDF5 compatibility).
- `S0`, `S1`, ...: plantar pressure sensor channels.
- `Ax`, `Ay`, `Az`: accelerometer axes.
- `Gx`, `Gy`, `Gz`: gyroscope axes.
- `lat`, `lng`: GPS coordinates (optional, depending on the trial).

## Data Lifecycle

1. **Source**: populated by the CLI `extract-data`, which queries InfluxDB
   using Flux and stores the results in HDF5 format.
```bash
   poetry run extract-data --config config/config.yaml
```
2. **Consumption**: consumed by the CLI `analyze-gait`, which reads a
   specific trial (identified by `analysis.h5_key` in `config.yaml`),
   computes gait features and writes them to `reports/`.
```bash
   poetry run analyze-gait --config config/config.yaml
```

## Integrity and Update Policy

- **Format**: binary HDF5 (PyTables `table` format with `data_columns=True`).
- **Append policy**: new trials are appended (`mode="a"`) to the existing
  file; existing keys are preserved.

## Git & Security Note

Raw data files (`.h5`, `.xlsx`, `.csv`, etc.) are excluded from version
control via `.gitignore`. To regenerate the local HDF5 database, ensure
valid InfluxDB credentials are available (via `${INFLUXDB_TOKEN}` or the
`config.yaml` file) and run the extraction CLI.

