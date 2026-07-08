Architecture
============

System Context
--------------

.. graphviz::

   digraph Context {
       rankdir=LR;
       node [shape=box, style=rounded];

       postgres [label="PostgreSQL\nhealthywear_event / healthywear_test_results"];
       influx [label="InfluxDB\nraw wearable streams"];
       config [label="Configuration\n.env + config/config.yaml"];
       extract [label="Extraction layer\nGaitDataExtractor"];
       process [label="Processing layer\nGaitDataProcessor"];
       cli [label="CLI orchestration\nextract-data / analyze-gait / run_batch.py"];
       storage [label="Local storage\nHDF5 + reports"];

       config -> cli;
       cli -> postgres;
       cli -> extract;
       extract -> influx;
       extract -> storage;
       cli -> process;
       storage -> process;
       process -> postgres;
   }

Module Responsibilities
-----------------------

- gait_analysis.extractor: selects and materializes raw trial data per foot.
- gait_analysis.processor: computes spatiotemporal, fatigue, bilateral, and spatial metrics.
- gait_analysis.postgresql: isolates database reads and writes behind typed repository models.
- gait_analysis.cli: exposes stable operational entry points for extraction and analysis.

Class Diagram
-------------

.. graphviz::

   digraph Classes {
       rankdir=TB;
       node [shape=record, style=rounded];

       influx_cfg [label="InfluxConfig|url:str|token:str|org:str|bucket:str"];
       process_cfg [label="ProcessConfig|fs:float|cutoff_pressure:float|cutoff_gyro:float|min_peak_distance_s:float|minute_block_duration_s:float|..."];
       postgres_cfg [label="PostgresConfig|host:str|user:str|password:str|database:str|port:int|schema:str|..."];
       selector [label="EventSelector|ids:list[int]|codeids:list[str]|test_types:list[str]|date_from:datetime|date_to:datetime"];
       extractor [label="GaitDataExtractor|run_batch_extraction()|run_batch_extraction_from_rows()|audit_registry_rows()|list_hdf5_keys()|close()"];
       processor [label="GaitDataProcessor|process_signals()|_detect_gait_events()|_compute_minute_metrics()|..."];
       repo [label="HealthywearPostgresRepository|fetch_events()|upsert_test_result()|upsert_test_results()"];

       extractor -> influx_cfg [label="validates"];
       processor -> process_cfg [label="uses"];
       repo -> postgres_cfg [label="validates"];
       repo -> selector [label="queries with"];
   }

Method Interaction Notes
------------------------

The runtime orchestration is intentionally thin: CLIs parse arguments, resolve
configuration, and delegate to typed domain services. This keeps the business
logic testable and allows documentation to map directly onto code boundaries.
