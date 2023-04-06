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
cdo'/home/nicolasf/mambaforge/envs/CDO/bin/cdo'


```


