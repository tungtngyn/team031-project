# Team 31 - Spring 2025

* Webapp is designed to run on Python 3.12 & the only libraries used are **pandas** and **bokeh** so far. Feel free to add more as needed.

* Download the CSVs / JSONs from Google Drive into the /data folder.
  
* When coding, the webapp is designed so you have access to `processed-data.csv` via `self.df` within any function in the AirfarePredictionApp class. Use this to build any visualizations / model inference needed.

* Class Convention: Within `AirfarePredictionApp`, functions starting with an underscore, e.g. `_`, denote it is meant to be used internal to the app itself. The only 'public' function is `.build` which is meant to be used externally and builds the webapp layout. 