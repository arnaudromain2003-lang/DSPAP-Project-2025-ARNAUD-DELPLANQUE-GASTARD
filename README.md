```
  _____   _____       _____        _____  
 |  __ \ / ____|     |  __ \ /\   |  __ \ 
 | |  | | (___ ______| |__) /  \  | |__) |
 | |  | |\___ \______|  ___/ /\ \ |  ___/ 
 | |__| |____) |     | |  / ____ \| |     
 |_____/|_____/   _  |_| /_/  _ \_\_|     
 |  __ \         (_)         | |          
 | |__) | __ ___  _  ___  ___| |_         
 |  ___/ '__/ _ \| |/ _ \/ __| __|        
 | |   | | | (_) | |  __/ (__| |_         
 |_|   |_|  \___/| |\___|\___|\__|        
                _/ |                      
               |__/```             

### Overview

> This repository contains the final project for the course Data Science – Principles and Applications (ENTPE).
> The project analyzes TCL public transport validation data (metro, tram, bus) aggregated at 15-minute intervals to:
> - extract typical daily mobility profiles,
> - detect abnormal days caused by disruptions or events
>
> **Authors: Romain Arnaud – Maxime Delplanque – Étienne Gastard**


 **Please extract data.zip directly into the data/ folder, making sure the extraction does not create a data/data/ subfolder. All files from the archive must end up directly inside data/.** 
 **https://filesender.renater.fr/?s=download&token=5eb5f9a4-64fc-4284-a772-481826a88f17**


### Installation guide

```git clone https://github.com/arnaudromain2003-lang/DSPAP-Project-2025-ARNAUD-DELPLANQUE-GASTARD.git

# Clone the repository

cd DSPAP-Project-2025 ('or any other folder name to contain our repository')

# Installation of the environnment

pip install -r requirements.txt

# or

conda env create -f dspap-ADG.yml
conda activate dspap```

### You can install the same environment as we used

conda create --name monenv --file environment.txt
pip install -r <(grep -A9999 "# Additional pip packages" environment.txt | tail -n +2)

### Dataset

We use three datasets from the Public Transport Validation Dataset (1 Nov 2019 to 30 Mar 2020):
- bus_indiv_15min.csv
- tramway_indiv_15min.csv
- subway_indiv_15min.csv (station-level flows)

**Please extract data.zip directly into the data/ folder, making sure the extraction does not create a data/data/ subfolder. All files from the archive must end up directly inside data/.**

### Essential variables:
- VAL_DATE (timestamp, 15-min resolution)
- Flow (bus/tram)
- Station flows (metro)


### Objectives

- Identify regular daily patterns across modes
- Perform K-means clustering on daily time series
- Detect anomalous mobility days


### Research question:

*How can TCL validation data be used to extract typical daily mobility profiles and highlight abnormal usage days?*


### Methods

Preprocessing: cleaning, time alignment, aggregation into daily 96-point vectors

Exploratory analysis: global patterns, weekday/weekend/holiday effects

Clustering: K-means on daily profiles (5–8 clusters)

Anomaly detection: cluster distance and temporal context

Spatial analysis: municipality-based aggregation for comparison

```### Project Structure
=================

DSPAP-Project (or your folder name)
└── notebooks/          # Analysis in Jupyter
    ├── Project_notebook.ipynb

└── src/                  # Core Python modules
    ├── clustering.py       # Clustering module : algorithm and plot functions
    ├── preprocessing.py    # Pre-processing of the data
    ├── plot.py             # Plot module
    ├── filters.py          # Module used to define pandas masks and to compute filtered dataframes
    └── df_operations,py    # Module to compute operations on pandas dataframes

└── data/               # Local datasets (ignored by git)        

 **Please extract data.zip directly into the data/ folder, making sure the extraction does not create a data/data/ subfolder. All files from the archive must end up directly inside data/.**

requirements.txt        # pip & conda environments
README.md               # Project documentation```
