# NIWA's Island Climate Update "Water Watch" 

Source code, scripts and notebooks for the NIWA Island Climate Update (ICU) "Water Watch": Drought monitoring and forecasting for the Southwest Pacific

## Requirements 

You can create a suitable conda environment by running: 

```
$ conda env create -f ICU.yml
$ conda activate ICU
$ pip install palettable
```

## Installation 

You should be able to install this pure Python package in the ICU environment using 

```
$ python setup.py install
```

## Background 

The aim of these modules, scripts and notebooks is to combine realtime rainfall monitoring in the Southwest Pacific, using the [NASA GPM-IMERG](https://gpm.nasa.gov/data/imerg) satellite product, and probabilistic monthly and seasonal forecast data (from 8 different General Circulation Models, or *GCMs*) to highlight regions that are and / or will be in potential "water-stress" conditions: i.e. - in a nutshell - where significant rainfall deficits have been recorded over the past few months, and the monthly or seasonal forecasts indicate high probability for dry conditions to persist or worsen. 

#### Data 

**1) The GPM-IMERG satellite rainfall estimates**

The past 30, 60, 90, 180 and 360 days rainfall accumulations, anomalies and percentile of scores are derived from the daily, [GPM-IMERG](https://gpm.nasa.gov/data/imerg) mission. The daily, near realtime (2 days lag) data is downloaded from https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.06 

The climatologies have been pre-computed from all the available data over the **2001 - 2020 period**. 

**3) Monthly and seasonal rainfall forecasts from the C3S Multi-Model Ensemble (MME)** 

Probabilistic forecasts are derived from a Multi-Model Ensemble (MME) including forecasts from the following 8 Coupled Ocean-Atmosphere GCMs: 

- ECMWF (SEAS5)
- UKMO (GloSea6-GC3.2)
- Meteo-France (Météo-France System 7)
- DWD (GCFS2.1)
- CMCC (CMCC-SPS3.5)
- NCEP (CFSv2)
- JMA (JMA/MRI-CPS2)
- ECCC (CanSIPSv2)

The details for the different GCM can be found [here](https://confluence.ecmwf.int/display/CKB/Description+of+the+C3S+seasonal+multi-system)

The forecast data (post 2017) for this MME contains in excess of 370 members.

The data is available from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/#!/home)

## Organisation of this repository

- [ICU_Water_Watch](https://github.com/nicolasfauchereau/ICU_Water_Watch/tree/main/ICU_Water_Watch): contains the source code itself, i.e. the collection of functions used for the data retrieval, processing, calculation and mapping of the various rainfall monitoring and forecasting products part of the ICU "Water Watch", the code is organized in 6 main modules: 

    - [utils.py](https://github.com/nicolasfauchereau/ICU_Water_Watch/blob/main/ICU_Water_Watch/utils.py): General utility functions 
    - [geo.py](https://github.com/nicolasfauchereau/ICU_Water_Watch/blob/main/ICU_Water_Watch/geo.py): Manipulation of geometries (from shapefiles), creation of masks, etc
    - [GPM.py](https://github.com/nicolasfauchereau/ICU_Water_Watch/blob/main/ICU_Water_Watch/GPM.py): Functions related to the download, processing and the calculation of diagnostics from the near-realtime GPM-IMERG data 
    - [C3S.py](https://github.com/nicolasfauchereau/ICU_Water_Watch/blob/main/ICU_Water_Watch/C3S.py): Functions related to the download, processing and derivation of probabilistic forecasts from the C3S Multi-Model Ensemble 
    - [verification.py](https://github.com/nicolasfauchereau/ICU_Water_Watch/blob/main/ICU_Water_Watch/verification.py): Functions related to the validation of the C3S individual GCMs and MME
    - [plot.py](https://github.com/nicolasfauchereau/ICU_Water_Watch/blob/main/ICU_Water_Watch/plot.py): Plotting and mapping functions 

- [notebooks](https://github.com/nicolasfauchereau/ICU_Water_Watch/tree/main/notebooks): contains all the notebooks, organized in 3 folders: [`GPM`](https://github.com/nicolasfauchereau/ICU_Water_Watch/tree/main/notebooks/GPM), [`C3S`](https://github.com/nicolasfauchereau/ICU_Water_Watch/tree/main/notebooks/C3S) and [`verification`](https://github.com/nicolasfauchereau/ICU_Water_Watch/tree/main/notebooks/verification), these notebooks illustrate the various processing steps, and also include the code used to calculate the various climatological quantities from the GPM-IMERG and C3S datasets.  

- [scripts](https://github.com/nicolasfauchereau/ICU_Water_Watch/tree/main/scripts): contains the python scripts, designed to be run from the command line, with keywords arguments, as part of the operational suite 

### Credits 

The development of this software was made possible by funding from NIWA's "core" (SSIF) funding under projects PRAS2101 and CAVA2101, as well as co-funding from MBIE project C01X1813. 

### Acknowledgments 

Thanks as well for support, feedbacks and advice from Doug Ramsay, Dr. Andrew Lorrey and Ben Noll from NIWA. 

### References 

 - Fauchereau N., Ramsay D., Lorrey A.M., Noll B.E (in preparation): Open data and open source software for the development of a multi-model ensemble monthly and seasonal forecast system for the Pacific region. To be submitted to *JOSS*, and preprint on *ESSOar*.  

 - Fauchereau N., Lorrey A.M, Noll B.E. (in preparation): On the predictability of New Zealand’s seasonal climate from General Circulation Models forecasts and observational indices. To be submitted to *Weather and Climate*.  

### Additional material and references