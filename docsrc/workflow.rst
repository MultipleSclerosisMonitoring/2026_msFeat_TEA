Pipeline Workflow
=================

Operational Stages
------------------

1. Event selection reads candidate tests from healthywear_event.
2. Extraction queries InfluxDB within the event time window and stores each foot in HDF5.
3. Analysis loads one HDF5 trial at a time, resamples signals, segments gait events, and computes metrics.
4. Bilateral enrichment combines left/right feet when both are available for the same test window.
5. Persistence upserts the final metrics into healthywear_test_results.

Processing Sequence
-------------------

.. graphviz::

   digraph Sequence {
       rankdir=LR;
       node [shape=record, style=rounded];

       cli_extract [label="CLI\nextract-data"];
       repo [label="HealthywearPostgresRepository"];
       extractor [label="GaitDataExtractor"];
       influx [label="InfluxDB"];
       hdf5 [label="HDF5 store"];
       cli_analyze [label="CLI\nanalyze-gait / run_batch"];
       processor [label="GaitDataProcessor"];
       results [label="healthywear_test_results"];

       cli_extract -> repo [label="fetch_events(selector)"];
       repo -> cli_extract [label="event rows"];
       cli_extract -> extractor [label="run_batch_extraction_from_rows()"];
       extractor -> influx [label="query_data_frame()"];
       influx -> extractor [label="raw gait dataframe"];
       extractor -> hdf5 [label="to_hdf(key per foot)"];
       cli_analyze -> hdf5 [label="read trial"];
       cli_analyze -> processor [label="process_signals()"];
       processor -> cli_analyze [label="metrics + events"];
       cli_analyze -> repo [label="upsert_test_result()"];
       repo -> results [label="INSERT ... ON CONFLICT DO UPDATE"];
   }

Metric Resolution Order
-----------------------

Spatial metrics are resolved in layers:

- fixed known-distance logic when the clinical protocol defines it;
- GPS when the test and data quality make it viable;
- IMU-based approaches such as imu_zupt when enabled;
- configured fallback models such as biometric or gyro_norm.

This layered approach ensures that spatial_method always records which
model produced the published distance, speed, and stride-length values.
