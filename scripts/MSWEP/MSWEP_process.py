#!/usr/bin/env python
# coding: utf-8

# %% imports 
import pathlib
import argparse

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from dask.diagnostics import ProgressBar

from ICU_Water_Watch import domains, utils, plot, geo, MSWEP 

# %% description 
parser = argparse.ArgumentParser(
    prog="MSWEP_process.py",
    description="""Put together the climatologies and the real time accumulations, and calculate the precipitation anomalies""",
)

parser.add_argument(
    "-i",
    "--ipath",
    type=str,
    default="/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge",
    help="""The path to the daily NRT MSWEP files\n
    default `/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge`""",
)

parser.add_argument(
    "-s",
    "--dpath_shapes",
    type=str,
    default="/home/nicolasf/operational/ICU/development/hotspots/data/shapefiles",
    help="""The path to shapefiles (ICU EEZs and coastlines)\n
    default `/home/nicolasf/operational/ICU/development/hotspots/data/shapefiles`""",
)

parser.add_argument(
    "-d",
    "--domain_name",
    type=str,
    default=f"SP",
    help="""The domain for which the MWSEP NRT files have been extracted\n
    default `SP` ('South Pacific': [100E, 240E, 50S, 30N])""",
)

parser.add_argument(
    "-n",
    "--ndays_agg",
    type=int,
    default=90,
    help="""The number of days for the rainfall accumulation, in [30,60,90,180,360]
    \ndefault 90 days""",
)

parser.add_argument(
    "-l",
    "--lag",
    type=int,
    default=2,
    help="""The lag to realtime, depending on when the script is run, must be between 1 and 2 days\n
    default 2 days""",
)

parser.add_argument(
    "-v",
    "--varname",
    type=str,
    default="precipitation",
    help="""The variable name\n
    default `precipitation`""",
)

parser.add_argument(
    "-cs",
    "--clim_start",
    type=int,
    default=1991,
    help="""The first year of the climatological period, can be 1991 or 1993\n
    default 1991""",
)

parser.add_argument(
    "-ce",
    "--clim_stop",
    type=int,
    default=2020,
    help="""The last year of the climatological period, can be 2020 or 2016\n
    default 2020""",
)

# %% parse the arguments
args = parser.parse_args()


# %% get the parameters
ipath = args.ipath
dpath_shapes = args.dpath_shapes
domain_name = args.domain_name
ndays_agg = args.ndays_agg
lag = args.lag
varname = args.varname
clim_start = args.clim_start
clim_stop = args.clim_stop 


# dpath = "/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge"
# dpath_shapes = "/home/nicolasf/operational/ICU/development/hotspots/data/shapefiles/"
# domain_name = "SP"
# nbdays_agg = 90
# lag_to_realtime = 1
# varname = "precipitation"
# fig_kwargs = dict(dpi=200, bbox_inches="tight", facecolor="w")
# clim_start = 1991
# clim_stop = 2020


dpath = pathlib.Path(ipath).joinpath(domain_name)

dpath_shapes = pathlib.Path(dpath_shapes)

dpath_climatology = dpath.joinpath(
    f"climatologies/{ndays_agg}days/{clim_start}_{clim_stop}/netcdf"
)

# %% output path 
opath = dpath.joinpath("outputs")
opath.mkdir(parents=True, exist_ok=True)
print(f"Output path for the merged realtime acccumulation and climatologies is {str(opath)}")

# %% get the EEZs shapefiles, individual + merged
EEZs, merged_EEZs = geo.get_EEZs(dpath_shapes=dpath_shapes)

# %%
today = datetime.utcnow().date()
date_stop = today - timedelta(days=lag)
DOY_stop = date_stop.timetuple().tm_yday
date_start = date_stop - timedelta(days=ndays_agg - 1)
DOY_start = date_start.timetuple().tm_yday


# %% lis the files
lfiles = list(dpath.glob("MSWEP_Daily_????-??-??.nc"))
lfiles.sort()
lfiles = lfiles[-ndays_agg:]

# %% get the bounding dates, from the list of files, and test
date_first_file = datetime.strptime(lfiles[0].name[-13:-3], "%Y-%m-%d").date()
date_last_file = datetime.strptime(lfiles[-1].name[-13:-3], "%Y-%m-%d").date()

# %% some tests 

if not(date_last_file == date_stop): 
    raise ValueError(f"The date for the last file ({date_last_file:%Y-%m-%d}) is different from the expected date ({date_stop:%Y-%m-%d})")

if not(date_first_file == date_start): 
    raise ValueError(f"The date for the first file ({date_first_file:%Y-%m-%d}) is different from the expected date ({date_start:%Y-%m-%d})")

# %% make the dataset from the list of file
dset = MSWEP.make_dataset(lfiles)

# %% get the attributes
last_date, ndays = MSWEP.get_attrs(dset)

# %% get the DOY from the lasst date 
DOY = last_date.timetuple().tm_yday

# %% get the right DOY for the climatology and correct in the file attributes
if (last_date.year % 4) == 0:
    print(f"{last_date.year} is a leap year, so DOY will go from {DOY} to {(DOY - 1)}")
    DOY -= 1

dset = dset.drop("DOY")
dset.attrs["DOY"] = DOY

# %% calculate the realtime accumulations
print(
    f"calculating accumulation for the {ndays} days period ending {last_date:%Y-%m-%d}\n"
)

dset_accum = MSWEP.calculate_realtime_accumulation(dset)

# %% rain days statistics
print(
    f"getting the rain days statistics for the {ndays} days period ending {last_date:%Y-%m-%d}\n"
)

dset_ndays = MSWEP.get_rain_days_stats(dset[[varname]], threshold=1)

# list the files for the climatologies 
lfiles_clim = list(dpath_climatology.glob(f"*DOY_{DOY:03d}*.nc"))

# %% should be 3 files
if len(lfiles_clim) != 3: 
    raise ValueError(f"There should be 3 for the climatologies, found {len(lfiles)}")

#%% open the climatological average
climatological_average = xr.open_dataset(
    dpath_climatology.joinpath(
        f"MSWEP_Daily_nogauge_DOY_{DOY:03d}_{ndays_agg}days_runsum_average.nc"
    )
)

# %% test the DOY 
DOY_clim = int(climatological_average["DOY"].data)
if DOY_clim != DOY:
    raise ValueError(f"The DOY in the climatology is {DOY_clim}, expected {DOY}")

# %% open the climatological quantiles 
climatological_quantiles = xr.open_dataset(
    dpath_climatology.joinpath(
        f"MSWEP_Daily_nogauge_DOY_{DOY:03d}_{ndays_agg}days_runsum_quantiles.nc"
    )
)

# %% test the DOY 
DOY_clim = int(climatological_quantiles["DOY"].data)
if DOY_clim != DOY:
    raise ValueError(f"The DOY in the climatology is {DOY_clim}, expected {DOY}")

# %% open the climatological SPI parameters 
climatological_SPI_params = xr.open_dataset(
    dpath_climatology.joinpath(
        f"MSWEP_Daily_nogauge_DOY_{DOY:03d}_{ndays_agg}days_runsum_SPI_params.nc"
    )
)

# %% test the DOY 
DOY_clim = int(climatological_SPI_params["DOY"].data)
if DOY_clim != DOY:
    raise ValueError(f"The DOY in the climatology is {DOY_clim}, expected {DOY}")

# %% remove the singleton dimension (DOY)
climatological_average = climatological_average.squeeze()
climatological_quantiles = climatological_quantiles.squeeze()
climatological_SPI_params = climatological_SPI_params.squeeze()

# %% rename the variable precipitation
climatological_average = climatological_average.rename({varname: f"{varname}_average"})

climatological_quantiles = climatological_quantiles.rename(
    {varname: f"{varname}_quantiles"}
)

# %% calculate the anomalies 
dset_accum["anoms"] = dset_accum[varname] - climatological_average[f"{varname}_average"]

# %% add the EEZs mask 
dset = geo.make_mask_from_gpd(dset, EEZs, subset=False, insert=True, mask_name="EEZs")

# %% merge the accumulations, anomalies, and the climatologies (average, quantiles and SPI parameters)
dset = xr.merge(
    (
        dset_accum,
        climatological_average,
        climatological_quantiles,
        climatological_SPI_params,
        dset_ndays
    )
)

# %% save to disk 
dset.to_netcdf(
    opath.joinpath(f"MSWEP_dset_merged_{ndays_agg}days_to_{last_date:%Y-%m-%d}.nc")
)