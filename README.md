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

The aim of these modules, scripts and notebooks is to combine realtime rainfall monitoring in the Southwest Pacific, using the NASA GPM-IMERG satellite product, and probabilistic monthly and seasonal forecast data (from 8 different General Circulation Models, or *GCMs*) to highlight regions that are are and / or will be in potential "water-stress" conditions: i.e. 

#### Data 

*1) The GPM-IMERG satellite rainfall estimates*

The past 30, 60, 90, 180 and 360 days rainfall accumulations, anomalies and percentile of scores are derived from the daily, [GPM-IMERG](https://gpm.nasa.gov/data/imerg) mission. The daily, near realtime (2 days lag) data is downloaded from https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.06 

The climatologies have been pre-computed from all the available data over the 2001 - 2019 period. 

*3) Monthly and seasonal rainfall forecasts from the C3S Multi-Model Ensemble (MME)* 

Probabilistic forecasts are derived from a Multi-Model Ensemble (MME) including forecasts from the following 8 General Circulation Models: 

- ECMWF 
- UKMO 
- Meteo-France 
- DWD 
- CMCC 
- NCEP 
- JMA 
- ECCC 

The forecast data (post 2017) for this MME contains in excess of 370 members.

The data is available from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/#!/home)

## Organisation of this repository

- [src](): contains the source code, i.e. the collection of functions used for the data retrieval, processing, calculation and mapping of the various rainfall monitoring and forecasting products part of the ICU "Water Watch", the code is organized in N modules: 

    - src/utils.py 
    - src/geo.py 
    - src/GPM.py 
    - src/C3S.py 
    - src/verification.py 

- [notebooks](): contains all the notebooks, organized in 3 folders: `GPM`, 'C3S` and `verification`  

- [scripts](): contains the python scripts, designed to be run from the command line, with keywords arguments, as part of the operational suite 

### Credits 

The development of this software was made possible by funding from NIWA's "core" (SSIF) funding under projects PRAS2101 and CAVA2101

### Acknowledgments 

Thanks as well for support, feedbacks and advice from Doug Ramsay, Dr. Andrew Lorrey and Ben Noll from NIWA. 

### References 

 - Fauchereau N., Ramsay D., Lorrey A.M., Noll B.E (in preparation): Open data and open source software for the development of a multi-model ensemble monthly and seasonal forecast system for the Pacific region. To be submitted to JOSS, and preprint on ESSOar.  

 - Fauchereau N., Lorrey A.M, Noll B.E. (in preparation): On the predictability of New Zealand’s seasonal climate from General Circulation Models forecasts and observational indices. To be submitted to Weather and Climate.  

### Additional material 
