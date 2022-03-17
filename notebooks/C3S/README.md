# running the notebooks for the download and processing of the C3S MME hindcasts / forecasts operationally 

## STEP 1

The first step (although step 1 and 2 can be run in any order really) is to download the *hindcast data (1993 - 2016)* for each of the current GCMs (in their latest system) from the Climate Data Store ([CDS](https://cds.climate.copernicus.eu/#!/home)), for all hindcasts initialised on the month corresponding to the current month. 

The notebook is **1_download_C3S_rolling_hindcasts.ipynb**

As we need the mapping between the different GCMs available on the CDS and the latest system available for the forecasts, this notebook relies on the YAML file named `CDS_config.yaml`. Currently (15 March 2022), this file looks like this: 

```
# configuration file, mapping GCM to system
ECMWF: 5
UKMO: 601
METEO_FRANCE: 8
DWD: 21
CMCC: 35
NCEP: 2
JMA: 3
ECCC_GEM_NEMO: 2
ECCC_CanCM4i: 3
```
The notebook is designed to be run via [papermill](https://papermill.readthedocs.io/en/latest/), the main parameters that one might want to change are: 

- the `lag` parameter, which allows you to specify the lag (in months) with respect to the current month, so that one can download hindcast data for initial months other than the current month, the default is 0.
- the `gcm_path` parameters, which points to the path where the hindcast datasets will be saved, the default is `/media/nicolasf/END19101/ICU/data/CDS/operational/hindcasts`
- the `config_yaml` parameter, which points to the path to the YAML file mapping GCM name to system 
  
So operationally, this notebook is simply run by calling: 

```
$ papermill 1_download_C3S_rolling_hindcasts.ipynb 1_download_C3S_rolling_hindcasts.ipynb 
```

## STEP 2

The second step is to download the latest *forecasts* from the 9 GCMs part of the C3S MME suite (see above) from the CDS. 

This is done by running the notebook **2_download_latest_C3S_forecasts.ipynb**.

Similarly to the **1_download_C3S_rolling_hindcasts.ipynb** notebook, it relies on the file `CDS_config.yaml`, and the parameters are the same, note that by default `gcm_path` points to `/media/nicolasf/END19101/ICU/data/CDS/operational/forecasts`

## STEP 3

The third step is to calculate the different lead-time dependent quantiles (terciles, quartiles, deciles, etc) climatologies for monthly and seasonal accumulation, from the updated hindcast data downloaded in step 1. This is done by running **3_GCMs_hindcast_climatology_ops.ipynb**, again using [papermill](https://papermill.readthedocs.io/en/latest/). The important parameters are: 

- `GCM`: The name of the GCM (needs to be in ['ECMWF','UKMO','METEO_FRANCE','CMCC','DWD', 'NCEP', 'JMA', 'ECCC_CanCM4i', 'ECCC_GEM_NEMO'])
- `period`: The accumulation period ('seasonal or 'monthly')
- `lag`: The lag (in months) with respect to the current month (again, so that one can calculates the retrospective climatologies)
- `gcm_path`: The path to the hindcast datasets, default is `/media/nicolasf/END19101/ICU/data/CDS/operational/hindcasts`
  
so typically, in an operational setting, this notebook is run like so:

```
for GCM in 'ECMWF' 'UKMO' 'METEO_FRANCE' 'CMCC' 'DWD' 'NCEP' 'JMA' 'ECCC_CanCM4i' 'ECCC_GEM_NEMO'; do 
    for period in 'monthly' 'seasonal'; do 
        papermill -p GCM ${GCM} -p period ${period} 3_GCMs_hindcast_climatology_ops.ipynb 3_GCMs_hindcast_climatology_ops.ipynb; 
    done; 
done; 
```

The climatologies will be saved in: 

`{gcm_path}/CLIMATOLOGY/{GCM}/TPRATE/`

## STEP 4

The fourth step is to calculate the latest forecasts probabilities, from the forecast data downloaded at step 2, and the climatologies calculated at step 3.

This is done by running the notebook **4_calculate_C3S_MME_forecast_probabilities.ipynb**.

The arguments (for papermill) are: 

- `GCM`: The name of the GCM (needs to be in ['ECMWF','UKMO','METEO_FRANCE','CMCC','DWD', 'NCEP', 'JMA', 'ECCC_CanCM4i', 'ECCC_GEM_NEMO'])
- `period`: The accumulation period ('seasonal or 'monthly')
- `lag`: The lag (in months) with respect to the current month (again, so that one can calculates the retrospective forecast probabilities)
- `gcm_path`: The path containing the hindcast *and* the forecast datasets, default is `/media/nicolasf/END19101/ICU/data/CDS/operational`
- `outputs_path`: The path where to save the netcdf files containing the probabilities, default is set currently to `/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/outputs/C3S`

so typically, in an operational setting, this notebook is run like so:

```
for GCM in 'ECMWF' 'UKMO' 'METEO_FRANCE' 'CMCC' 'DWD' 'NCEP' 'JMA' 'ECCC_CanCM4i' 'ECCC_GEM_NEMO'; do 
    for period in 'monthly' 'seasonal'; do 
        papermill -p GCM ${GCM} -p period ${period} 4_calculate_C3S_MME_forecast_probabilities.ipynb 4_calculate_C3S_MME_forecast_probabilities.ipynb; 
    done; 
done; 
```

## STEPS 5,6,7

This is where we plot the maps of monthly or seasonal tercile, decile probabilities as well as the probability for precipitation being below the 25th percentile (1st quartile). 

There are 3 notebooks that one needs to run: 

- *5_map_C3S_MME_probabilistic_tercile_forecast.ipynb*
- *6_map_C3S_MME_probabilistic_decile_forecast.ipynb*
- *7_map_C3S_MME_probabilistic_1st_quartile_forecast.ipynb*

Typically, this is how these are run operationally 

For the monthly tercile probabilistic forecasts:

```
for lead in 1 2 3 4 5; do 
    papermill -p period "monthly" -p lead ${lead} 5_map_C3S_MME_probabilistic_tercile_forecast.ipynb 5_map_C3S_MME_probabilistic_tercile_forecast.ipynb;
done; 
```

For the seasonal tercile probabilistic forecasts:

```
for lead in 1 2 3; do 
    papermill -p period "seasonal" -p lead ${lead} 5_map_C3S_MME_probabilistic_tercile_forecast.ipynb 5_map_C3S_MME_probabilistic_tercile_forecast.ipynb;
done; 
```

For the monthly decile probabilistic forecasts (i.e. the first decile category where the cumulative probability reaches 50%)

```
for lead in 1 2 3 4 5; do 
    papermill -p period "monthly" -p lead ${lead} 6_map_C3S_MME_probabilistic_decile_forecast.ipynb 6_map_C3S_MME_probabilistic_decile_forecast.ipynb; 
done; 
```

For the seasonal decile probabilistic forecasts:

```
for lead in 1 2 3; do 
    papermill -p period "seasonal" -p lead ${lead} 6_map_C3S_MME_probabilistic_decile_forecast.ipynb 6_map_C3S_MME_probabilistic_decile_forecast.ipynb; 
done; 
```

For the monthly probability for precipitation accumulations being below the 25th percentile (st quartile): 

```
for lead in 1 2 3 4 5; do 
    papermill -p period "monthly" -p lead ${lead} 7_map_C3S_MME_probabilistic_1st_quartile_forecast.ipynb 7_map_C3S_MME_probabilistic_1st_quartile_forecast.ipynb; 
done; 
```

For the seasonal probability for precipitation accumulations being below the 25th percentile:

```
for lead in 1 2 3; do 
    papermill -p period "seasonal" -p lead ${lead} 7_map_C3S_MME_probabilistic_1st_quartile_forecast.ipynb 7_map_C3S_MME_probabilistic_1st_quartile_forecast.ipynb; 
done; 
```

## STEP 6

The final step is to derive and map the ICU "Water Stress Outlook" which combines GPM-IMERG realtime data (the percentiles of scores for the past 90 days accumulation) and the probabilistic forecasts from the C3S MME.

This is done by running the notebook **6_map_ICU_Water_Stress_outlook.ipynb**. Note that currently it needs to be run twice: Once with setting the parameter `period` (at the beginning of the notebook) to "monthly", and once to "seasonal".