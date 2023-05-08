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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

from ICU_Water_Watch import domains, utils, plot, geo, MSWEP 

# %% small function to digitize
def _digitize(x, bins):
    return np.digitize(x.ravel(), bins.ravel())

# %% description 
parser = argparse.ArgumentParser(
    prog="MSWEP_map.py",
    description="""calculate the EAR, USDM and SPI categories, and map them as well as precipitation
    accumulations, anomalies, dry days and days since last rain""",
)

parser.add_argument(
    "-i",
    "--ipath",
    type=str,
    default="/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge",
    help="""The path to the daily merged MSWEP 2.8.0 netcdf files\n
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
    "-f",
    "--fig_path",
    type=str,
    default="/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/figures",
    help="""The path to save the figures\n
    default `/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/figures`""",
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
    help="""The cycle time i.e. `%Y-%m-%d`\n
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
fig_path = args.fig_path
domain_name = args.domain_name
ndays_agg = args.ndays_agg
cycle_time = args.cycle_time
varname = args.varname
clim_start = args.clim_start
clim_stop = args.clim_stop

# %% 
# ipath = "/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge"
# dpath_shapes = "/home/nicolasf/operational/ICU/development/hotspots/data/shapefiles"
# fig_path = "/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/figures"
# domain_name = "SP"
# ndays_agg = 90
# varname = "precipitation"
# clim_start = 1991
# clim_stop = 2020

# %% keyword arguments for the figures
fig_kwargs = dict(dpi=200, bbox_inches="tight", facecolor="w")

# %% paths to pathlib.Path
dpath = pathlib.Path(ipath).joinpath(f"{domain_name}/outputs")
dpath_shapes = pathlib.Path(dpath_shapes)

# %% paths to pathlib.Path
dpath = pathlib.Path(ipath).joinpath(f"{domain_name}/outputs")
dpath_shapes = pathlib.Path(dpath_shapes)

# %% 
fig_path = pathlib.Path(fig_path)
fig_path.mkdir(parents=True, exist_ok=True)

print(f"The figures will be saved in {str(fig_path)}")

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


# %% get the EEZs and coastlines 
EEZs, merged_EEZs = geo.get_EEZs(dpath_shapes=dpath_shapes)
coastlines = geo.get_coastlines(dpath_shapes=dpath_shapes)

# %% get the file containing the data for the past N days +  the climatologies 
fname = dpath.joinpath(f"MSWEP_dset_merged_{ndays_agg}days_to_{cycle_time:%Y-%m-%d}.nc")

#%% test if the file is on disk 
if not(fname.exists()): 
    raise ValueError(f"MSWEP_dset_merged_{ndays_agg}days_to_{cycle_time:%Y-%m-%d}.nc does not exist")

#%% open the file, with dask chunking 
dset = xr.open_dataset(fname, chunks={"time": "auto", "lat": 50, "lon": 50})

#%% get the attributes
attrs = {
    k: dset.attrs[k]
    for k in [
        "ndays",
        "last_day",
        "DOY",
    ]
}

# %% get the date from the time coordinate
dset_date = dset["time"]
dset_date = pd.Timestamp(dset_date.to_numpy()[0])

# %% remove the time singleton dimension 
dset = dset.squeeze()

# %% insert the mask 
dset = geo.make_mask_from_gpd(
    dset, merged_EEZs, subset=False, insert=True, mask_name="EEZs"
)

# %% plot the precip accumulations
plot.map_precip_accum(
    dset,
    varname=varname,
    mask="EEZs",
    geoms=EEZs,
    source="MSWEP 2.8.0",
    fpath=fig_path,
    close=True,
)

# %% plot the precip anomalies
plot.map_precip_anoms(
    dset,
    varname="anoms",
    mask="EEZs",
    geoms=EEZs,
    source="MSWEP 2.8.0",
    fpath=fig_path,
    close=True,
)

# %% plot the number of dry days
plot.map_dry_days_Pacific(
    dset,
    varname="dry_days",
    mask="EEZs",
    geoms=EEZs,
    source="MSWEP 2.8.0",
    fpath=fig_path,
    close=True,
)

# %% plot the number of days since last rain
plot.map_days_since_rain_Pacific(
    dset,
    varname="days_since_rain",
    mask="EEZs",
    geoms=EEZs,
    source="MSWEP 2.8.0",
    fpath=fig_path,
    close=True,
)

# %% now define the quantile thresholds for the EAR Watch Categories
EAR_threshs = [0.05, 0.1, 0.25, 0.9]

EAR_quantiles = dset[[f"{varname}_quantiles"]].sel(quantile=EAR_threshs)

EAR_colors_list = ["#F04E37", "#F99D1C", "#FFDE40", "#FFFFFF", "#33BBED"]

EAR_labels = [
    "Severely dry (< 5%)",
    "Seriously dry (< 10%)",
    "Warning (< 25%)",
    "Near or Wetter",
    "Seriously wet (> 90%)",
]

# %% derive the EAR categories 
EAR_categories = xr.apply_ufunc(
    _digitize,
    dset[varname],
    EAR_quantiles[f"{varname}_quantiles"],
    input_core_dims=[[], ["quantile"]],
    vectorize=True,
    dask="parallelized",
)

print(f"calculating the EAR Watch categories for {ndays_agg} accumulations ending {cycle_time:%Y-%m-%d}")
with ProgressBar():
    EAR_categories = EAR_categories.compute()

# %% add back the attributes 
EAR_categories.attrs = attrs

# % plot 
title = f'Water Stress (aligned to "EAR" alert levels), source MSWEP 2.8.0\n{ndays_agg} days to {cycle_time:%d %b %Y}'

plot.map_categories(
    EAR_categories,
    mask=dset["EEZs"],
    colors_list=EAR_colors_list,
    labels_list=EAR_labels,
    geoms=EEZs,
    extent=domains.domains["Water_Watch"],
    gridlines=False,
    title=title,
    figname_root="EAR",
    fpath=fig_path,
    close=True,
)

# %% now define the quantile thresholds for the USDM (US Drought Monitor) Categories
USDM_threshs = [0.02, 0.05, 0.1, 0.2, 0.3]

USDM_quantiles = dset[["precipitation_quantiles"]].sel(quantile=USDM_threshs)

USDM_colors_list = ["#8a0606", "#fc0b03", "#fc9003", "#ffd08a", "#ffeb0f", "#ffffff"]

USDM_labels = [
    "D4 (Exceptional Drought)",
    "D3 (Extreme Drought)",
    "D2 (Severe Drought)",
    "D1 (Moderate Drought)",
    "D0 (Abnormally Dry)",
    "None",
]

# %% derive the USDM categories 
USDM_categories = xr.apply_ufunc(
    _digitize,
    dset[varname],
    USDM_quantiles[f"{varname}_quantiles"],
    input_core_dims=[[], ["quantile"]],
    vectorize=True,
    dask="parallelized",
)

print(f"calculating the USDM categories for {ndays_agg} accumulations ending {cycle_time:%Y-%m-%d}")
with ProgressBar():
    USDM_categories = USDM_categories.compute()

# %% add back the attributes
USDM_categories.attrs = attrs


# %% plot 
title = f"US Drought Monitor (USDM), source MSWEP 2.8.0\n{ndays_agg} days to {cycle_time:%d %b %Y}"

plot.map_categories(
    USDM_categories,
    mask=dset["EEZs"],
    colors_list=USDM_colors_list,
    labels_list=USDM_labels,
    geoms=EEZs,
    extent=domains.domains["Water_Watch"],
    gridlines=False,
    title=title,
    figname_root="USDM",
    fpath=fig_path,
    cbar_yanchor=0.55,
    close=True
)

# %% now calculate the SPI (Standardized Precipitation Index)
SPI = MSWEP.calculate_SPI(dset[varname], dset["alpha"], dset["beta"], name="SPI")

# %% SPI thresholds
SPI_threshs = [-2, -1.5, -1, 1, 1.5, 2]

# %% derive the SPI categories
print(f"calculating the SPI categories for {ndays_agg} accumulations ending {cycle_time:%Y-%m-%d}")

SPI_categories = np.digitize(SPI["SPI"].data, SPI_threshs)

SPI["SPI_categories"] = (("lat", "lon"), SPI_categories)

# %% add the attributes 
SPI["SPI_categories"].attrs = attrs

# %% colors list and labels for the SPI 
SPI_colors_list = ["#F04E37", "#F99D1C", "#FFDE40", "#FFFFFF", "#96ceff", "#4553bf", "#09146b"]

SPI_labels = [
    "Extremely dry",
    "Severely dry",
    "Moderately dry",
    "Near normal",
    "Moderately wet",
    "Severely wet",
    "Extremely wet",
]

# %% plot 
title = f"Standardized Precipitation Index (SPI), source MSWEP 2.8.0\n{ndays_agg} days to {cycle_time:%d %b %Y}"

plot.map_categories(
    SPI["SPI_categories"],
    mask=dset["EEZs"],
    colors_list=SPI_colors_list,
    labels_list=SPI_labels,
    geoms=EEZs,
    extent=domains.domains["Water_Watch"],
    gridlines=False,
    spacing={"lon": 20, "lat": 10},
    title=title,
    cbar_yanchor=0.495,
    cbar_xanchor=0.85,
    figname_root="SPI",
    fpath=fig_path,
    close=True,
)

# %% Now start some data munging before plotting the country level maps

EAR_categories_ds = EAR_categories.to_dataset(name="EAR_categories")
USDM_categories_ds = USDM_categories.to_dataset(name='USDM_categories')
SPI_categories_ds = SPI[['SPI_categories']]


# %% main loop 
for country_name in coastlines.country_na.values:

    print(f"Now mapping the country level EAR, USDM and SPI for {country_name}")

    country_fname = utils.sanitize_name(country_name)
    
    EEZ = EEZs.query(f"COUNTRYNAM == '{country_name}'")

    coastline = coastlines.query(f"country_na == '{country_name}'")
    
    SPI_categories_subset = geo.make_mask_from_gpd(SPI_categories_ds, EEZ, subset=True, mask_name='mask_EEZ', domain_buffer=0.2)
    EAR_categories_subset = geo.make_mask_from_gpd(EAR_categories_ds, EEZ, subset=True, mask_name='mask_EEZ', domain_buffer=0.2)
    USDM_categories_subset = geo.make_mask_from_gpd(USDM_categories_ds, EEZ, subset=True, mask_name='mask_EEZ', domain_buffer=0.2)
    
    
    plot.map_categories(
    SPI_categories_subset['SPI_categories'],
    mask=SPI_categories_subset["mask_EEZ"],
    colors_list=SPI_colors_list,
    labels_list=SPI_labels,
    geoms=[EEZ, coastline],
    extent=domains.get_domain(SPI_categories_subset),
    gridlines=True,
    cartopy_coastlines=False,
    spacing={"lon": 2.5, "lat": 2.5},
    title=f"SPI, {country_name}, {ndays_agg} days to {cycle_time:%Y-%m-%d}",
    figname_root=f"SPI_{country_fname}", 
    fpath=fig_path,
    cbar_xanchor=1.01,
    cbar_yanchor=0,
    title_top=True, 
    fit=True,
    close=True,
    )

    plot.map_categories(
    EAR_categories_subset['EAR_categories'],
    mask=EAR_categories_subset["mask_EEZ"],
    colors_list=EAR_colors_list,
    labels_list=EAR_labels,
    geoms=[EEZ, coastline],
    extent=domains.get_domain(SPI_categories_subset),
    gridlines=True,
    cartopy_coastlines=False,
    spacing={"lon": 2.5, "lat": 2.5},
    title=f"EAR, {country_name}, {ndays_agg} days to {cycle_time:%Y-%m-%d}",
    figname_root=f"EAR_{country_fname}", 
    fpath=fig_path,
    cbar_xanchor=1.01,
    cbar_yanchor=0,
    title_top=True, 
    fit=True,
    close=True,
    )
    
    plot.map_categories(
    USDM_categories_subset['USDM_categories'],
    mask=USDM_categories_subset["mask_EEZ"],
    colors_list=USDM_colors_list,
    labels_list=USDM_labels,
    geoms=[EEZ, coastline],
    extent=domains.get_domain(SPI_categories_subset),
    gridlines=True,
    cartopy_coastlines=False,
    spacing={"lon": 2.5, "lat": 2.5},
    title=f"USDM, {country_name}, {ndays_agg} days to {cycle_time:%Y-%m-%d}",
    figname_root=f"USDM_{country_fname}", 
    fpath=fig_path,
    cbar_xanchor=1.01,
    cbar_yanchor=0,
    title_top=True, 
    fit=True,
    close=True,
    )