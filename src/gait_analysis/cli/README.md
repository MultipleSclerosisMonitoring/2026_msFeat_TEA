# CLI Interface

## Overview

This module provides the command-line interface (CLI) for executing the MS-Feat pipeline.

It acts as the entry point for users, connecting external configuration files with the core processing modules. The CLI enables reproducible execution of the pipeline without modifying the source code.


## Available Commands

### extract-data

Extracts sensor data from InfluxDB and stores it locally in HDF5 format.
`extract-data --config config/config.yaml`

Selective extraction and audit examples:

`extract-data --config config/config.yaml --ids 87 89`

`extract-data --config config/config.yaml --codeid RHRHUG004-1 --test 6MWT`

`extract-data --config config/config.yaml --ids 87 89 --check-only`

`extract-data --config config/config.yaml --test 6MWT --missing-only`

`extract-data --config config/config.yaml --list-hdf5-keys`

Typical targeted workflow:

1. Check whether the selected CSV rows are already present in HDF5:

`extract-data --config config/config.yaml --ids 87 89 --check-only`

2. Extract only the desired cases:

`extract-data --config config/config.yaml --ids 87 89`

3. Inspect the available HDF5 keys if needed:

`extract-data --config config/config.yaml --list-hdf5-keys`

4. Analyze a specific extracted foot without editing the config file:

`analyze-gait --config config/config.yaml --h5-key p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Right`

### analyze-gait

Processes extracted data and computes gait-related metrics.
`analyze-gait --config config/config.yaml`

You can override the configured HDF5 key directly:

`analyze-gait --config config/config.yaml --h5-key p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Right`


## Common Arguments

Both commands support the following arguments:

- `--config`  
  Path to the configuration file

- `--verbose`  
  Verbosity level (0=errors, 1=info, 2=details, 3=debug)

- `--lang`  
  Language for messages (if enabled)

- `--h5-key`
  Optional override for the HDF5 key to analyze

### extract-data specific arguments

- `--ids`
  Extract or audit only the specified CSV row ids

- `--codeid`
  Extract or audit only the specified patient identifiers

- `--missing-only`
  Skip rows already complete in HDF5 (both `Left` and `Right`)

- `--check-only`
  Compare the selected CSV rows against the HDF5 and exit without extracting

- `--list-hdf5-keys`
  Print the HDF5 keys currently available and exit


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
- Supports selective extraction and HDF5 audit workflows
