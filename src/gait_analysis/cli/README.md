# CLI Interface

Entry points for the MS-Feat pipeline. Thin orchestration layer
that delegates all logic to `gait_analysis` core modules.

## Commands

### extract-data
Extracts data from InfluxDB and stores it in HDF5.

```bash
poetry run extract-data --config config/config.yaml

# Selective
poetry run extract-data --config config/config.yaml --ids 87 89
poetry run extract-data --config config/config.yaml --codeid RHRHUG004-1 --test 6MWT

# Audit
poetry run extract-data --config config/config.yaml --check-only
poetry run extract-data --config config/config.yaml --missing-only
poetry run extract-data --config config/config.yaml --list-hdf5-keys
```

### analyze-gait
Processes one HDF5 trial and outputs metrics + plots.
Automatically loads the contralateral foot and computes bilateral metrics.

```bash
poetry run analyze-gait --config config/config.yaml
poetry run analyze-gait --config config/config.yaml \
  --h5-key p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Left
poetry run analyze-gait --config config/config.yaml --no-plots
```

## Arguments

| Argument | Commands | Description |
|---|---|---|
| `--config` | both | Path to config YAML |
| `--verbose` | both | 0=errors 1=info 2=details 3=debug |
| `--lang` | both | Message language (es/en) |
| `--h5-key` | analyze-gait | Override HDF5 key to analyze |
| `--no-plots` | analyze-gait | Skip plot generation |
| `--ids` | extract-data | CSV row ids to extract |
| `--codeid` | extract-data | Patient identifier filter |
| `--test` | extract-data | Test type filter (6MWT/TUG/T25FW) |
| `--missing-only` | extract-data | Skip rows already in HDF5 |
| `--check-only` | extract-data | Audit without extracting |
| `--list-hdf5-keys` | extract-data | Print available keys and exit |

## Output (analyze-gait)
- `reports/data/metrics_summary_<key>.csv` — trial-wide metrics
- `reports/data/metrics_per_minute_<key>.csv` — per-block fatigue metrics
- `reports/plots/gait_segmentation_<key>.png` — S2 signal with HS/TO
- `reports/plots/fatigue_analysis_<key>.png` — stride time and cadence by block
