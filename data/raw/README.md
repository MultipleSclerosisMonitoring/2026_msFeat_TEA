# Data Storage - Raw Signals

This directory contains the local persistent storage for the gait analysis trials.

## HDF5 Database (`gait_study_data.h5`)
This file serves as the **central repository** and the bridge between the Extraction stage and the Analysis stage.

### Structure
The data is organized hierarchically to optimize access speeds:
- `/p_[SubjectID]/[TestType]/trial_[Index]`

### Data Lifecycle
1. **Source**: Populated by `scripts/batch_extractor.py` from InfluxDB.
2. **Consumption**: Read by `scripts/batch_process_all.py` for feature extraction.

## Data Integrity
- **Format**: Binary HDF5 (Compressed).
- **Update Policy**: New trials are appended to the existing file; existing keys are preserved to avoid data loss.