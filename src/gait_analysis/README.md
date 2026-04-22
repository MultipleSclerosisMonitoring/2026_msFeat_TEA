# gait_analysis: Core Module

## Overview

This package contains the core implementation of the MS-Feat framework.

It provides the main logic for data extraction, signal processing, and computation of gait-related features. The module is designed to be independent of the user interface and fully driven by external configuration.


## Structure

The module is organized into the following components:

### extractor.py

Responsible for:

- Connecting to InfluxDB
- Querying high-frequency sensor data
- Validating data structure
- Storing extracted data in HDF5 format


### processor.py

Implements:

- Signal filtering and preprocessing
- Gravity compensation
- Detection of gait events (Heel Strike, Toe Off)
- Computation of spatiotemporal gait parameters
- Estimation of fatigue metrics based on temporal trends


## Responsibilities

This module is responsible for:

- Domain-specific processing logic
- Data transformation and analysis
- Implementation of biomechanical algorithms


## Non-responsibilities

This module does NOT handle:

- Command-line interfaces
- Argument parsing
- User interaction
- Execution workflows

These responsibilities are implemented in:
`gait_analysis.cli`


## Execution

This module is not intended to be executed directly.

All functionality is exposed through the CLI layer:
`extract-data`, `analyze-gait`

## Design Principles

- Separation of concerns between core logic and interface
- Configuration-driven execution
- Reusability and modularity
- Suitability for batch processing workflows