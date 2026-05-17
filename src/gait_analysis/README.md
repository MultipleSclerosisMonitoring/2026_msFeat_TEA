# gait_analysis: Core Package

Core logic of the MS-Feat framework. Fully driven by external
configuration; no hardcoded paths or parameters.

## Modules

### extractor.py
- Connects to InfluxDB and queries sensor time-series
- Validates required columns (S2, _time, Foot) via Pydantic
- Persists data to HDF5 with hierarchical keys:
  `p_<codeid>/<test>/start_<timestamp>/<foot>`

### processor.py
Public API:
- `GaitDataProcessor.process_signals()` — full pipeline for one trial
- `compute_bilateral_metrics()` — asymmetry and double support from L+R
- `compute_gps_path_metrics()` — GPS path length via haversine
- `compute_mean_swing_gyro_integral()` — gyro-norm spatial estimator
- `haversine_m()` — great-circle distance between two GPS points

Pipeline stages inside `process_signals()`:
1. Chronological sorting and uniform resampling (100 Hz)
2. Automatic vertical axis detection (gravity-based)
3. Zero-phase Butterworth filtering (S2: 20 Hz, gyro: 5 Hz)
4. Turn detection via L2 gyroscope norm
5. Heel Strike and Toe-Off detection (valley + threshold / derivative)
6. Temporal metrics: stride time, cadence, stance, swing, CV
7. Fatigue analysis: linear slopes over 60-second blocks
8. Spatial metrics: known distance / GPS / gyro-norm / biometric model
9. Bilateral fusion: asymmetry and double support (via CLI)

## Design principles
- Stateless: no memory between trials
- Reproducible: same HDF5 + config.yaml → identical output
- Orientation-agnostic: L2 norm for turn detection, variance for axis calibration