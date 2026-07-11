Command-Line Interface
======================

Extraction CLI
--------------

Use extract-data to select curated events and materialize raw sensor data
into the local HDF5 store.

.. code-block:: bash

   poetry run extract-data --config config/config.yaml --source postgres --ids 552

Common selectors:

.. code-block:: bash

   poetry run extract-data --config config/config.yaml --codeid EPSHUG067-10
   poetry run extract-data --config config/config.yaml --test 6MWT --from-date 2026-01-01 --to-date 2026-03-31
   poetry run extract-data --config config/config.yaml --ids 552 --check-only
   poetry run extract-data --config config/config.yaml --missing-only

Analysis CLI
------------

Use analyze-gait to process one extracted HDF5 key, compute metrics, and
optionally persist the results when the PostgreSQL configuration is available.

.. code-block:: bash

   poetry run analyze-gait --config config/config.yaml \n     --h5-key p_EPSHUG067-10/6MWT/start_2026-01-22T11-00-00Z/Right

Batch analysis can be orchestrated by scripts/run_batch.py when both feet
and multiple events must be processed in one run.

Direct Batch CLI
----------------

Use run-direct-batch when you want to execute the full operational flow in one
pass, without materializing intermediate HDF5 datasets. The command selects
curated events, queries InfluxDB, processes each available foot in memory,
writes a batch CSV, and optionally upserts the results into PostgreSQL.

.. code-block:: bash

   poetry run run-direct-batch --config config/config.yaml --source postgres --ids 66 67 68
   poetry run run-direct-batch --config config/config.yaml --source postgres --ids 66 67 68 --no-postgres-persist
   poetry run run-direct-batch --config config/config.yaml --test-type 6MWT --from-date 2026-01-01 --to-date 2026-03-31

Operational Guidance
--------------------

- Prefer --source postgres for production runs so the event registry stays authoritative.
- Use --check-only before large extractions to audit HDF5 completeness.
- Keep pipeline_version updated whenever the metric semantics change.
- Reprocessing the same event-foot pair updates the stored row instead of creating duplicates.
