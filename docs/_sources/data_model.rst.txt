Data Model
==========

Source Tables
-------------

healthywear_event stores the curated test windows used to drive extraction.
The pipeline expects at least the following fields:

- id: unique event identifier.
- codeid: patient code.
- t_code: standardized test type such as 6MWT or TUG.
- d_from: event start timestamp.
- d_until: event end timestamp.

Result Table Semantics
----------------------

healthywear_test_results stores one row per event and foot.
The natural identity is (healthywear_event_id, foot). Reprocessing the same
test updates the existing row in place through PostgreSQL upsert semantics.

Result Object View
------------------

.. graphviz::

   digraph ResultObject {
       rankdir=TB;
       node [shape=record, style=rounded];

       result [label="TestResult|healthywear_event_id:int|patient_id:str|test_type:str|foot:str|pipeline_version:str|analysis_h5_key:str|analysis_timestamp:datetime"];
       temporal [label="Temporal metrics|stride_time_*|stance_time_*|swing_time_*|cadence_*|minute slopes"];
       spatial [label="Spatial metrics|spatial_method|spatial_distance_m|walking_speed_mean_m_s|stride_length_mean_m"];
       bilateral [label="Bilateral metrics|step-time asymmetry|double support|availability flag"];
       processing [label="Processing provenance|fs|cutoff_pressure|cutoff_gyro|thresholds|minute block duration"];

       result -> temporal;
       result -> spatial;
       result -> bilateral;
       result -> processing;
   }

Persistence Contract
--------------------

The repository layer converts computed metric dictionaries into a database-ready
payload. Only keys present in the documented result-column map are persisted.
This prevents accidental schema drift and makes the write contract explicit.
