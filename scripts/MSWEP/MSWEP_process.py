#!/usr/bin/env python
# coding: utf-8

# %% imports 
import pathlib
import argparse

from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

from ICU_Water_Watch import geo, MSWEP 

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
    "-c",
    "--cycle_time",
    type=str,
    default=None,
    help="""The cycle time i.e YYYY-MM-DD\n
    default None""",
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
cycle_time = args.cycle_time
varname = args.varname
clim_start = args.clim_start
clim_stop = args.clim_stop 

# %% values for testing 
# ipath = "/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge"
# dpath_shapes = "/home/nicolasf/operational/ICU/development/hotspots/data/shapefiles/"
# domain_name = "SP"
# ndays_agg = 90
# varname = "precipitation"
# fig_kwargs = dict(dpi=200, bbox_inches="tight", facecolor="w")
# clim_start = 1991
# clim_stop = 2020

# %% paths to pathlib.Path
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

# %% cycle time 

if cycle_time is not None: 

    cycle_time = datetime.strptime(cycle_time, "%Y-%m-%d").date()

else: 
    # if not defined, we revert to 2 days lag to realtime

    cycle_time = datetime.utcnow().date() - relativedelta(days=2)

# %% check that the cycle time is not in the future 
today = datetime.utcnow().date()

if cycle_time > today: 
    raise ValueError(f"cycle_time is set to {cycle_time:%Y-%m-%d}, but today (UTC) is {today:%Y-%m-%d}")

DOY_cycle_time = cycle_time.timetuple().tm_yday

# %% list the files on disk 
lfiles = list(dpath.glob("MSWEP_Daily_????-??-??.nc"))
lfiles.sort()
lfiles = lfiles[-ndays_agg:]

# %% get the intended start date for the accumulation 
date_start = cycle_time - relativedelta(days = ndays_agg - 1)

# %% build the list of dates for the files that should be on disk 
ldates = pd.date_range(date_start, cycle_time, freq='D')

# %% build the list of files that should be on disk 
lfiles_from_cycle_time = [
    dpath.joinpath(f"MSWEP_Daily_{dt:%Y-%m-%d}.nc") for dt in ldates
]

# print("\n".join([f.name for f in lfiles_from_cycle_time]))
# print("\n".join([f.name for f in lfiles]))

# %% compare the list of files on disk and the list of files that *should* be on disk 
lfiles_intersection = list(set(lfiles_from_cycle_time) & set(lfiles))
lfiles_intersection.sort()


# %% the length of the intersection should be equal to the number of days for the accumulation 
if len(lfiles_intersection) != ndays_agg: 
    lfiles_missing = list(set(lfiles_from_cycle_time) - set(lfiles))
    lfiles_missing = [str(f.name) for f in lfiles_missing]
    lfiles_missing = '\n'.join(lfiles_missing)
    message = f"""\nThe number of available files on disk for cycle time {cycle_time:%Y-%m-%d} ({len(lfiles_intersection)}) is not equal to the number of days ({ndays_agg})\n
    files missing = {lfiles_missing}"""
    raise ValueError(message)


# %% get the bounding dates, from the list of files, and test
date_first_file = datetime.strptime(lfiles_from_cycle_time[0].name[-13:-3], "%Y-%m-%d").date()
date_last_file = datetime.strptime(lfiles_from_cycle_time[-1].name[-13:-3], "%Y-%m-%d").date()

# %% make the dataset from the list of file
dset = MSWEP.make_dataset(lfiles_from_cycle_time)

# %% 
dset = dset.chunk({'time':-1, 'lat':100, 'lon':100})

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

# %% rename the variable precipitation to precipitation_average
climatological_average = climatological_average.rename({varname: f"{varname}_average"})

# %% rename 
climatological_quantiles = climatological_quantiles.rename(
    {varname: f"{varname}_quantiles"}
)

# %% calculate the anomalies 

climatological_average = climatological_average.chunk({'lat':100, 'lon':100})

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

dset = dset.chunk({'time':-1, 'quantile':-1, 'lat':100, 'lon':100})

# %% save to disk 
print(f"saving MSWEP_dset_merged_{ndays_agg}days_to_{last_date:%Y-%m-%d}.nc in {str(opath)}")
with ProgressBar():
    dset.to_netcdf(
        opath.joinpath(f"MSWEP_dset_merged_{ndays_agg}days_to_{last_date:%Y-%m-%d}.nc")
    )

dset.close()

# %% EOF
