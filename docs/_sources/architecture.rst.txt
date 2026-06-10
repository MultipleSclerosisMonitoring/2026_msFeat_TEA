Architecture
============

The project is organized around a modular batch-processing pipeline that separates
data extraction, signal processing, command-line execution, and reporting.

.. graphviz::

   digraph G {
       rankdir=LR;
       node [shape=box, style=rounded];

       InfluxDB [label="InfluxDB"];
       Config [label="config/config.yaml"];
       ExtractCLI [label="CLI: extract-data"];
       AnalyzeCLI [label="CLI: analyze-gait"];
       Extractor [label="gait_analysis.extractor"];
       Processor [label="gait_analysis.processor"];
       HDF5 [label="HDF5 dataset"];
       Reports [label="reports/metrics_summary.csv"];

       Config -> ExtractCLI;
       Config -> AnalyzeCLI;
       InfluxDB -> ExtractCLI;
       ExtractCLI -> Extractor;
       Extractor -> HDF5;
       HDF5 -> AnalyzeCLI;
       AnalyzeCLI -> Processor;
       Processor -> Reports;
   }