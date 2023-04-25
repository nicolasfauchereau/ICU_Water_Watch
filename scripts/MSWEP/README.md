# Processing of the MSWEP daily 'nogauge' dataset 

### 1) extraction of the Pacific domain 

The first step is to extract the Pacific domain from the original global daily netcdf files. 

This is handled by the script `extract_daily_MSWEP_nogauge.sh`   

In addition to the extraction itself, which is done using the [sellonlatbox](https://code.mpimet.mpg.de/projects/cdo/embedded/index.html#x1-1900002.3.5) operator of [CDO](https://code.mpimet.mpg.de/projects/cdo/), this bash script also calls `compress_netcdf.py` which uses xarray to read the extracted files, and compress them 
using the `encodings` parameter of the `to_netcdf` method, see [the xarray documentation](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html). 

### 2) merging the daily files into yearly files 

We use [CDO](https://code.mpimet.mpg.de/projects/cdo/) for this, so in the the directory where the extracted (and compressed) daily files reside, do 

```
year_start=1979
year_stop=2020
cdo='/home/nicolasf/mambaforge/envs/CDO/bin/cdo'

for y in {${year_start}..${year_end}}; do ${cdo} mergetime ${y}???_comp.nc merged_${y}.nc; echo "processing ${y}"; done
```

### 3) compressing (again) the merged yearly files

```
year_start=1979
year_stop=2020
compress='/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/scripts/MSWEP/compress_netcdf.py'

for y in {${year_start}..${year_end}}; do ${compress} merged_${y}.nc; echo "processing ${y}"; done

```

### 4) merge the yearly files into one single file covering the whole period

```
year_start=1979
year_stop=2020
cdo='/home/nicolasf/mambaforge/envs/CDO/bin/cdo'

${cdo} mergetime merged_????_comp.nc merged_{year_start}_{year_stop}.nc
```


### 5) compress the merged file

```
compress='/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/scripts/MSWEP/compress_netcdf.py'

${compress} merged_{year_start}_{year_stop}.nc
```

### 6) delete the leap days

```
year_start=1979
year_stop=2020
cdo='/home/nicolasf/mambaforge/envs/CDO/bin/cdo'

${cdo} del29feb merged_{year_start}_{year_stop}_comp.nc merged_{year_start}_{year_stop}_comp_noleap.nc
```

### 7) calculate the running N days accumulations

```
year_start=1979
year_stop=2020
ndays = 30
cdo='/home/nicolasf/mambaforge/envs/CDO/bin/cdo'

mkdir ${ndays}days

${cdo} --timestat_date last runsum,${ndays} merged_{year_start}_{year_stop}_comp_noleap.nc ${ndays}days/merged_{year_start}_{year_stop}_chunked_noleap_runsum_{ndays}.nc
```

### 8) for each DOY, extract a window, and export dataset to ZARR, prior to calculating the climatologies

see help of `extract_daily_MSWEP_runsum_DOY_with_buffer_to_zarr.py`: 


```
usage: extract_daily_MSWEP_runsum_DOY_with_buffer_zarr.py [-h] [-n NDAYS_AGG] [-s DOY_START] [-e DOY_STOP] [-f FNAME] [-o OPATH] [-w WINDOW]

construct zarr datasets for each DOY from a netcdf file containing running accumulations

optional arguments:
  -h, --help            show this help message and exit
  -n NDAYS_AGG, --ndays_agg NDAYS_AGG
                        The number of days for the rainfall accumulation, in [30,60,90,180,360] default 90 days
  -s DOY_START, --doy_start DOY_START
                        The day of year to start the loop default 1
  -e DOY_STOP, --doy_stop DOY_STOP
                        The day of year to stop the loop default 365
  -f FNAME, --fname FNAME
                        The input filename, e.g. 'merged_1979_2020_chunked_noleap_runsum_{ndays_agg}.nc' no default
  -o OPATH, --opath OPATH
                        The directory root where to save the zarr datasets, a 'climatologies/${ndays_agg}days' directory will be created therein default
                        /media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/SP
  -w WINDOW, --window WINDOW
                        The size of the window in days, should be an odd number, the buffer is defined as {window}//2 default is 15 (7 days each side of the target DOY)
```

### 9) calculate the climatological quantities (average, percentiles, SPI alpha and gamma parameters) from the DOY zarr datasets

see help of `calculate_climatologies_MSWEP_nogauge_window_construct_from_zarr.py`: 

```
usage: calculate_climatologies_MSWEP_nogauge_window_construct_from_zarr.py [-h] [-n NDAYS_AGG] [-s DOY_START] [-e DOY_STOP] [-cs CLIM_START] [-ce CLIM_STOP] [-v VARNAME]
                                                                      [-i IPATH_ZARR] [-d DASK_DIR]

calculate the MSWEP climatologies (average, quantiles, SPI alpha and beta params) from the buffered DOY ZARR files for the 'no-gauge' version of Daily MSWEP, see
`extract_daily_MSWEP_runsum_DOY_with_buffer_to_zarr.py` for the processing of the running accumulation and the creation of the ZARR files containing the data varying by
year and buffer (7 days each side) for each day of year

optional arguments:
  -h, --help            show this help message and exit
  -n NDAYS_AGG, --ndays_agg NDAYS_AGG
                        The number of days for the rainfall accumulation, in [30,60,90,180,360] default 90 days
  -s DOY_START, --doy_start DOY_START
                        The day of year to start the loop\ndefault 1
  -e DOY_STOP, --doy_stop DOY_STOP
                        The day of year to stop the loop default 365
  -cs CLIM_START, --clim_start CLIM_START
                        The start year for the climatological period default 1993
  -ce CLIM_STOP, --clim_stop CLIM_STOP
                        The end year for the climatological period default 2016
  -v VARNAME, --varname VARNAME
                        The variable name default 'precipitation'
  -i IPATH_ZARR, --ipath_zarr IPATH_ZARR
                        The path containing the zarr datasets (1 for each DOY) default `/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/SP/climatologies/`
  -d DASK_DIR, --dask_dir DASK_DIR
                        The path to the dask folder, default `./dask_dir`
```





