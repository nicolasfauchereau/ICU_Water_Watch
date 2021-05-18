# NIWA's Island Climate Update "Water Watch" 

Source code, scripts and notebooks for the NIWA Island Climate Update (ICU) "Water Watch": Drought monitoring and forecasting for the Southwest Pacific

## Requirements 

You can create a suitable conda environment by running: 

```
$ conda env create -f ICU.yml
$ conda activate ICU
$ pip install palettable
```

## Background 

The aim of these modules, scripts and notebooks is to combine realtime rainfall monitoring in the Southwest Pacific, using the NASA GPM-IMERG satellite product, and probabilistic monthly and seasonal forecast data (from 6 GCMs) to highlight regions that are are and / or will be in potential "water-stress" conditions: i.e. 

#### Data 

1) The GPM-IMERG satellite rainfall estimates 

The past 30, 60, 90, 180 and 360 days rainfall accumulations, anomalies and percentile of scores are derived from the daily, [GPM-IMERG](https://gpm.nasa.gov/data/imerg) mission. The daily, near realtime (2 days lag) data is downloaded from https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.06 

The climatologies have been pre-computed from all the available data over the 2001 - 2019 period. 

3) Monthly and seasonal rainfall forecasts from the CDS Multi-Model Ensemble (MME)

Probabilistic forecasts are derived from a Multi-Model Ensemble (MME) including forecasts from the following General Circulation Models: 

- ECMWF 
- UKMO 
- Meteo-France 
- DWD 
- CMCC 
- NCEP 

The forecast data (post 2017) for this MME contains between 

## Organisation of this repository

- [src](): contains the source code, i.e. the collection of functions used for the data retrieval, processing, calculation and mapping of the various rainfall monitoring and forecasting products part of the ICU "Water Watch", the code is organized in N modules: 

    - src/utils.py 
    - src/geo.py 
    - src/GPM.py 
    - src/C3S.py 
    - src/verification.py 

- [notebooks](): contains all the notebooks 

- [scripts](): contains the python scripts, designed to be run from the command line, with keywords arguments, as part of the operational suite 

### Pre-requesites 

### Dependencies 

### Credits 

The development of this library was made possible by funding from NIWA's "core" (SSIF) funding 

### Acknowledgments 

Thanks as well for support, feedbacks and advice from Doug Ramsay and Dr. Andrew Lorrey

### Reference 

### Additional material 