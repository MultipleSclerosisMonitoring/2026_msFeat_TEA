# Configuration

## Overview

The MS-Feat pipeline is fully driven by external configuration files.

All runtime parameters are defined in YAML format, which ensures reproducibility, flexibility, and separation between code and execution settings.


## Files

### config.yaml

Main configuration file used during execution.

It defines:

- InfluxDB connection settings
- Input and output paths
- Signal processing parameters
- Logging verbosity
- Language settings
- Execution options


### config.example.yaml

Template configuration file versioned in the repository.

This file provides a safe starting point and must be copied before running the pipeline.

## Usage

Create a local configuration file from the template:

```bash
cp config/config.example.yaml config/config.yaml
cp .env.example .env
```

The CLI automatically loads `.env` from the project root and resolves
`${VAR_NAME}` placeholders referenced inside `config/config.yaml`.

## Design Principles

- No hardcoded paths in code
- Full separation between configuration and implementation
- Reproducible experiments
- Environment-independent execution


## Execution

The configuration file is used by both CLI commands:
```bash
poetry run extract-data --config config/config.yaml
poetry run analyze-gait --config config/config.yaml
```

## Notes
- `.env` may contain sensitive information and should not be versioned
- config.example.yaml is the version intended to be shared in the repository
- All runtime behavior should be controlled from configuration rather than modifying the source code
