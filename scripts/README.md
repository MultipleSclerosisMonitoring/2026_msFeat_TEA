# Legacy Scripts

## Overview

This directory contains **legacy and auxiliary scripts** from earlier stages of
the MS-Feat project. They are kept for historical reference and traceability of
the development process.

**These scripts are not part of the current pipeline.** The official execution
path goes through the CLI entry points provided by the `gait_analysis` package:

```bash
poetry run extract-data --config config/config.yaml
poetry run analyze-gait --config config/config.yaml
```

## Status

| Script | Status | Notes |
|---|---|---|
| `process_gait_signals.py` | Deprecated | Predecessor of the `analyze-gait` CLI. Replaced by `gait_analysis.cli.analyze_gait`. |
| `batch_process_all.py` | Deprecated | Batch analysis engine from an earlier iteration. Functionality partially absorbed into the current CLI. |
| `batch_extractor.py` | Obsolete | Early CLI wrapper for the extractor. Incompatible with the current `GaitDataExtractor` API. |
| `diagnostico_db.py` | Obsolete | One-off InfluxDB schema exploration tool. Paths and configuration file no longer reflect the current project layout. |

## Why they are kept

- **Traceability**: they document the evolution of the pipeline.
- **Reproducibility of early results**: some intermediate artifacts were
  generated with these scripts.
- **Reference**: they contain alternative implementation ideas that may be
  useful in the future.

## Why they are not used

- They predate the installable package structure (`src/gait_analysis/`).
- They rely on APIs that have since been refactored.
- The current pipeline is fully driven by external YAML configuration and
  exposed through CLI entry points defined in `pyproject.toml`.