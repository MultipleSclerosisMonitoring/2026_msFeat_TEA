Project Modules
===============

The production package lives under src/gait_analysis and is organized into
cohesive modules with explicit runtime roles.

- gait_analysis.extractor: bulk extraction from InfluxDB into HDF5.
- gait_analysis.processor: signal processing and metric computation.
- gait_analysis.postgresql: event selection and result persistence.
- gait_analysis.cli.extract_data: extraction-oriented command-line entry point.
- gait_analysis.cli.analyze_gait: single-trial analysis entry point.

The API reference page expands these modules with autodocumented classes,
functions, signatures, and type hints.
