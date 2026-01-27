Introduction
============

The main purpose of this package is to help in the indetification of Gait features.
The ambition is to characterize the fine grain attributes of the walking process
in particular for people with Multiple Sclerosis (MS) disease.

Motivation
==========

The focus is to seamless integrated the work carried out at edge, by collecting high
frequency data obtained from Sensoria heath \copyright instrumented socks via BLE 
from Android or iOS developed app to upload it into a timeseries database in InfluxDB

In addition, other tools analyze the collected data and extract relevant periods where
data is available and store them into a RDBMS in PostgreSQL. Indeed, through different 
algorithms, periods larger than 7s of walk are also identified.

Now, it is the time to identify detailed low level gait features, such as the swing distance,
cadence, and others.


General Structure
=================

The application includes:

* One package named ``InfluxDBms`` containing several classes for managing and querying InfluxDB.
* One module named ``test_InfluxDB`` with several scripts to extract and present data from InfluxDB.
* One module named ``gps_trajectory`` for generating and visualizing GPS trajectories using Folium.

Detailed information can be found in the modules below.
