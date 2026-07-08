MS-Feat Documentation
=====================

.. raw:: html

   <div class="doc-hero">
     <div class="doc-hero__logos">
       <img src="_static/upm-logo.png" alt="Universidad Politecnica de Madrid logo" class="doc-hero__logo doc-hero__logo--upm">
       <img src="_static/etsii-logo.png" alt="ETSII UPM logo" class="doc-hero__logo doc-hero__logo--etsii">
     </div>
     <div class="doc-hero__text">
       <p class="doc-hero__eyebrow">ETSII / UPM Research Documentation</p>
       <p class="doc-hero__lead">MS-Feat is a clinical gait-analysis pipeline that extracts raw wearable signals from InfluxDB, aligns them to curated test windows stored in PostgreSQL, computes spatiotemporal gait metrics per test and foot, and persists the results for downstream clinical review.</p>
     </div>
   </div>

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
