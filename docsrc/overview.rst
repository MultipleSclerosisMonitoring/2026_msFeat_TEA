Project Overview
================

Purpose
-------

MS-Feat supports standardized gait-test processing in a reproducible pipeline.
It is intended for scenarios where:

- test windows are curated externally and stored in healthywear_event;
- raw signals come from wearable IMU and plantar-pressure devices in InfluxDB;
- analyses must be repeatable, configurable, and traceable by pipeline version;
- results must be available per clinical test and per foot.

Core Capabilities
-----------------

- Select trials from PostgreSQL by event identifier, patient code, test type, or date window.
- Extract left/right sensor streams from InfluxDB and store them in HDF5.
- Compute temporal, fatigue-related, bilateral, and spatial gait indicators.
- Persist computed outputs to healthywear_test_results using upsert semantics.
- Keep configuration externalized via config/config.yaml and environment variables.

Design Principles
-----------------

- Strongly typed runtime configuration through Pydantic models.
- Deterministic processing with explicit fallback strategies for spatial metrics.
- Separation between selection, extraction, processing, persistence, and CLI orchestration.
- Documentation-first engineering with API reference, architecture views, and UML diagrams.

Documentation Coverage
----------------------

This documentation includes:

- architectural flow diagrams;
- sequence and class UML-style diagrams rendered through Graphviz;
- CLI usage guidance;
- data-model and persistence notes;
- autodocumented API pages enriched with type hints and Google-style docstrings.
