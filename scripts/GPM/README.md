# How to run the GPM-IMERG download, processing and mapping script 

You can call the help on these script by (e.g.)

```
$ GPM_process.py --help
```

## GPM_update.py 

Download missing files on disk from https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.06/, interpolate to the TRMM grid and saves in netcdf

```
usage: GPM_update.py [-h] [-d DPATH]

optional arguments:
  -h, --help            show this help message and exit
  -d DPATH, --dpath DPATH
                        the path where to save the GPM-IMERG realtime netcdf files, REQUIRED
```

## GPM_process.py 

Process the data, calculates the rainfall accumulation, percentiles of scores, and dry and wet days statistics 

```
usage: GPM_process.py [-h] [-d DPATH] [-n NDAYS] [-l LAG] [-ds DPATH_SHAPES] [-o OPATH]

optional arguments:
  -h, --help            show this help message and exit
  -d DPATH, --dpath DPATH
                        the path where to find the GPM-IMERG realtime netcdf files and climatologies, REQUIRED
  -n NDAYS, --ndays NDAYS
                        the number of days over which to calculate the accumulation and take the climatology
  -l LAG, --lag LAG     the lag (in days) to realtime, if run in the morning (NZ time) 1 should be OK
  -ds DPATH_SHAPES, --dpath_shapes DPATH_SHAPES
                        the path to the `EEZs` and `Coastlines` folder containing the respective shapefiles, REQUIRED
  -o OPATH, --opath OPATH
                        the path where to save the outputs files (netcdf and geotiff), REQUIRED
```

## GPM_map.py 

Does all the mapping and creates the figures 

```
usage: GPM_map.py [-h] [-d DPATH] [-n NDAYS] [-ds DPATH_SHAPES] [-f FPATH]

optional arguments:
  -h, --help            show this help message and exit
  -d DPATH, --dpath DPATH
                        the path where to find the netcdf files containing the accumulations and the dry days statistics, REQUIRED
  -n NDAYS, --ndays NDAYS
                        the number of days for the above
  -ds DPATH_SHAPES, --dpath_shapes DPATH_SHAPES
                        the path to the `EEZs` and `Coastlines` folder containing the respective shapefiles, REQUIRED
  -f FPATH, --fpath FPATH
                        the path where to save the figures, REQUIRED

```
