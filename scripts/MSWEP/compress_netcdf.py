#!/home/nicolasf/mambaforge/envs/ICU_ops/bin/python

import sys
import os
import pathlib
import xarray as xr

fname = sys.argv[1]

fname = pathlib.Path(fname)

name = fname.name
dpath = fname.parent

varname = 'precipitation'

encoding={varname:{'zlib':True, 'shuffle':True, 'complevel':1}}

dset = xr.open_dataset(fname)

# get the dimensions

dset.to_netcdf(dpath.joinpath(name.replace('.nc','') + '_comp.nc'), encoding=encoding, format='NETCDF4')

os.remove(str(fname))

print(f"processed {fname}")