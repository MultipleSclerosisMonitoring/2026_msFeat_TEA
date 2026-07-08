MS-Feat Documentation
=====================

MS-Feat is a clinical gait-analysis pipeline that extracts raw wearable signals
from InfluxDB, aligns them to curated test windows stored in PostgreSQL,
computes spatiotemporal gait metrics per test and foot, and persists the
results for downstream clinical review.

The documentation is written in English, generated with Sphinx, and designed
to serve both as developer reference and as a deployment-ready knowledge base
for GitHub Pages.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   architecture
   workflow
   data_model
   cli
   api
   modules
