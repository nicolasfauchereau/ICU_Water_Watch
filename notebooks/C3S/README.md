# running the notebooks for the download and processing of the C3S MME hindcasts / forecasts operationally 

## 1) forecast maps and ICU "Water Stress Outlook" 

#### STEP 1

The first step (although step 1 and 2 can be run in any order really) is to download the *hindcast data (1993 - 2016)* for each of the current GCMs (in their latest system) from the Climate Data Store (CDS), for reforecast initialised on the current month. Currently, this and all other notebooks are parametrized for the following GCMs: 'ECMWF','UKMO','METEO_FRANCE','CMCC','DWD', 'NCEP', 'JMA', 'ECCC_CanCM4i', 'ECCC_GEM_NEMO' (the latter 2 being part of the ECCC 2 model ensemble).

The download of the hindcast dataset is done by running the notebook **1_download_C3S_rolling_hindcasts.ipynb**. 

#### STEP 2

The second step is to download the latest *forecasts* from the 9 GCMs part of the C3S MME suite (see above) from the CDS. 

This is done by running the notebook **2_download_latest_C3S_forecasts.ipynb**.

#### STEP 3

The third step is to calculate / update the climatologies, using the updated hindcast data downloaded in step 1. This is done by running **3_drive_C3S_GCMs_hindcast_climatology_ops.ipynb**. This notebook uses [papermill](https://papermill.readthedocs.io/en/latest/) to loop over list of parameters (GCMs, and monthly and seasonal accumulation) and actually calls **GCMs_hindcast_climatology_ops.ipynb**, which does the heavy lifting. Any error at execution will be reflected (with cell number) in the latter. Note that by default **3_drive_C3S_GCMs_hindcast_climatology_ops.ipynb** / **GCMs_hindcast_climatology_ops.ipynb** calculates the climatologies for the hindcasts initialised on the month corresponding to the current month, but there is a `lag` parameter (expressed in months) if calculations for previous months need to be performed. 

#### STEP 4

The fourth step is to calculate the latest forecasts probabilities, from the forecast data downloaded at step 2, and the climatologies calculated above.

This is done by running the notebook **4_calculate_C3S_MME_forecast_probabilities.ipynb**.

#### STEP 5

The fifth step is the mapping of the tercile, decile probabilistic forecasts and the probability for rainfall being below the 25th percentile. 

This is done by running the notebook **5_drive_forecast_maps.ipynb**. This notebook uses papermill to send parameters and run 3 other notebooks: 

- **5_map_C3S_MME_probabilistic_tercile_forecast.ipynb** (makes the tercile probabilities maps)
- **5_map_C3S_MME_probabilistic_decile_forecast.ipynb** (makes the cumulative decile probabilities maps)
- **5_map_C3S_MME_probabilistic_percentile_forecast.ipynb** (map the probability for rainfall being below the 25th percentile)

#### STEP 6

The final step is to derive and map the ICU "Water Stress Outlook" which combines GPM-IMERG realtime data (the percentiles of scores for the past 90 days accumulation) and the probabilistic forecasts from the C3S MME.

This is done by running the notebook **6_map_ICU_Water_Stress_outlook.ipynb**. Note that currently it needs to be run twice: Once with setting the parameter `period` (at the beginning of the notebook) to "monthly", and once to "seasonal"