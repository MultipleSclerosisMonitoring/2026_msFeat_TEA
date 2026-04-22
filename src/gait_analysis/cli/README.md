# CLI Interface

## Overview

This module provides the command-line interface (CLI) for executing the MS-Feat pipeline.

It acts as the entry point for users, connecting external configuration files with the core processing modules. The CLI enables reproducible execution of the pipeline without modifying the source code.


## Available Commands

### extract-data

Extracts sensor data from InfluxDB and stores it locally in HDF5 format.
`extract-data --config config/config.yaml`

### analyze-gait

Processes extracted data and computes gait-related metrics.
`analyze-gait --config config/config.yaml`


## Common Arguments

Both commands support the following arguments:

- `--config`  
  Path to the configuration file

- `--verbose`  
  Verbosity level (0=errors, 1=info, 2=details, 3=debug)

- `--lang`  
  Language for messages (if enabled)


## Usage

Commands must be executed within the Poetry environment:

```bash
poetry run extract-data --config config/config.yaml
poetry run analyze-gait --config config/config.yaml
```

## Design
- Thin orchestration layer
- Delegates all processing logic to gait_analysis core modules
- Fully driven by external configuration
- Ensures reproducible execution across environments
- Designed for batch processing workflows