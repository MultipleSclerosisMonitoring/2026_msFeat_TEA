# Configuration

All runtime parameters are defined in YAML. No hardcoded values in code.

## Files

| File | Purpose |
|---|---|
| `config.example.yaml` | Versioned template — copy this to start |
| `config.yaml` | Local config — not versioned, never commit |
| `.env.example` | Versioned secrets template |
| `.env` | Local secrets — never commit |

## Setup

```bash
cp config/config.example.yaml config/config.yaml
cp .env.example .env
# Edit config.yaml and .env with your values
```

## Sections

### project
Language (`es`/`en`) and verbosity level (0–3).

### paths
Input HDF5 path, registry CSV path, output metrics and plot paths.

### processing
Signal processing parameters: sampling frequency, filter cutoffs,
gyroscope threshold, peak detection settings, toe-off method.

### analysis
Default HDF5 key used by `analyze-gait` when `--h5-key` is not provided.

### influxdb / postgresql
Connection settings. All values must be set as `${VAR_NAME}` resolved
from `.env`. Never write credentials directly in the YAML.

### clinical_tests
Protocol distances and plausibility thresholds for TUG and T25FW:
`distance_m`, `min_duration_s`, `min_strides`.

### gps_estimation
Quality filters for GPS-based spatial estimation (6MWT outdoor):
`min_span_m`, `min_unique_points`.

### spatial_models
Constants and enable flags for the three inertial spatial estimators:
- `gyro_norm.K` — gyroscope integral model (RMSE ~21%)
- `biometric.K` — cadence regression model (RMSE ~11%)
- `imu_zupt.enabled` — Madgwick + ZUPT (disabled until ±8g recalibration)