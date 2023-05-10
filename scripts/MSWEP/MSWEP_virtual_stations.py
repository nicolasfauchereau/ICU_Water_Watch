#!/usr/bin/env python
# coding: utf-8

# %% 
import pathlib
import argparse 
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

from dask.diagnostics import ProgressBar

from ICU_Water_Watch import utils


# %% description 
parser = argparse.ArgumentParser(
    prog="MSWEP_virtual_stations.py",
    description="""Extract 'virtual stations' from MSWEP and plots time-series and climatological accumulations for a given accumulation period 
    (30, 60, 90, 180, 360 days)""",
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
    "-o",
    "--opath",
    type=str,
    default="/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge",
    help="""The path where to save the EAR, USDM and SPI categories\n
    default `/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge{domain}/outputs`""",
)

parser.add_argument(
    "-s",
    "--stations",
    type=str,
    default="./Stations_for_extractions.csv",
    help="""The path to the CSV file with the station information (station_name,country,lat,lon)\n
    default `./Stations_for_extractions.csv`""",
)


parser.add_argument(
    "-f",
    "--fig_path",
    type=str,
    default="/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/figures/virtual_stations",
    help="""The path to save the figures\n
    default `/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/figures/virtual_stations`""",
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
    help="""The cycle time (YYYY-MM-DD)\n
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

# %% parse arguments 
ipath = args.ipath
opath = args.opath
fig_path = args.fig_path
stations = args.stations
domain_name = args.domain_name
ndays_agg = args.ndays_agg
cycle_time = args.cycle_time
varname = args.varname
clim_start = args.clim_start
clim_stop = args.clim_stop

# %% 
# domain = "SP"
# ndays_agg = 90
# varname = 'precipitation'
# cycle_time = '2023-05-08'
# ipath = "/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/"
# opath = "/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/"
# fig_path = "/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/figures/virtual_stations/"
# stations = "../../data/Stations/Stations_for_extractions.csv"
# clim_start = 1991
# clim_stop = 2020

# %% figures kwargs 
fig_kwargs = dict(dpi=200, bbox_inches="tight", facecolor="w")

# %% string to pathlib.Path 

ipath = pathlib.Path(ipath).joinpath(domain_name)

opath = pathlib.Path(opath)

if opath == ipath: 
    
    opath = opath.joinpath('outputs/virtual_stations')
    
    opath.mkdir(parents=True, exist_ok=True)

fig_path =  pathlib.Path(fig_path)

fig_path.mkdir(parents=True, exist_ok=True)

clim_path = ipath.joinpath(f'climatologies/{ndays_agg}days/{clim_start}_{clim_stop}/netcdf')

# %% cycle time 
cycle_time = datetime.strptime(cycle_time, "%Y-%m-%d")

# %% DOY manip 
DOY = cycle_time.timetuple().tm_yday

if (cycle_time.year % 4) == 0:
    print(f"\n{cycle_time.year} is a leap year, so DOY will go from {DOY} to {(DOY - 1)}\n")
    DOY -= 1

# %%  read the climatological averages, quantiles, SPI parameters 
clim_ave = xr.open_dataset(clim_path.joinpath(f"MSWEP_Daily_nogauge_DOY_{DOY:03d}_{ndays_agg}days_runsum_average.nc")) 
clim_quantiles = xr.open_dataset(clim_path.joinpath(f"MSWEP_Daily_nogauge_DOY_{DOY:03d}_{ndays_agg}days_runsum_quantiles.nc")) 
clim_SPI = xr.open_dataset(clim_path.joinpath(f"MSWEP_Daily_nogauge_DOY_{DOY:03d}_{ndays_agg}days_runsum_SPI_params.nc")) 

# %% read the stations coordinates 
station_coords = pd.read_csv(stations, index_col=None)

station_coords = station_coords.dropna()

# %% NRT (extracted for the domain) MSWEp nogauge files 

lfiles_netcdf = list(ipath.glob("MSWEP_Daily_????-??-??.nc"))

lfiles_netcdf.sort()

# %% lons and lats to extract 
lons_to_extract = station_coords.lon.to_xarray()
lats_to_extract = station_coords.lat.to_xarray()

lons_to_extract = lons_to_extract.rename({"index":"station_name"})
lats_to_extract = lats_to_extract.rename({"index":"station_name"})

# ID (station name + country)
stname_country = list(map(", ".join, zip(station_coords.station_name.values.tolist(), station_coords.country.values.tolist())))

# %% add the correct coordinates and dimension names 
lons_to_extract['station_name'] = (('station_name'), stname_country)
lats_to_extract['station_name'] = (('station_name'), stname_country)

# %% get the list of files up to cycle time 

index_stop = lfiles_netcdf.index(ipath.joinpath(f"MSWEP_Daily_{cycle_time:%Y-%m-%d}.nc"))

lfiles = lfiles_netcdf[index_stop - ndays_agg + 1 :index_stop + 1]

if len(lfiles) != ndays_agg: 
    raise ValueError(f"the number of files ({len(lfiles)}) does not match the number of days passed as argument ({ndays_agg})")

# %% open the multiple files dataset in parallel, with chunks only over the time dimension, as 
# we do spatial extractions 
dset = xr.open_mfdataset(lfiles, parallel=True, chunks={'time':1, 'lon':-1, 'lat':-1})

# %% extract the points 
dset_stations = dset[[varname]].sel(lon=lons_to_extract, lat=lats_to_extract, method="nearest")

print(f"\nstarting extraction now, for past {ndays_agg} days to {cycle_time:%Y-%m-%d}\n")

with ProgressBar(): 
    
    dset_stations = dset_stations.compute()

# %% cast to a pandas dataframe 
dset_stations_df = dset_stations[varname].to_pandas()

# %% now extract the stations coordinates from the climatologies

clim_ave = clim_ave.sel(lon=lons_to_extract, lat=lats_to_extract, method="nearest")
clim_quantiles = clim_quantiles.sel(lon=lons_to_extract, lat=lats_to_extract, method="nearest")
clim_SPI = clim_SPI.sel(lon=lons_to_extract, lat=lats_to_extract, method="nearest")

# %% average {ndays_agg} accumulation up to that DOY
clim_ave = clim_ave[varname].to_pandas()

# %% accumulation for the realtime time-series 
dset_stations_df_sum = dset_stations_df.sum(0)


# %% data munging 
clim_ave = clim_ave.T 
clim_ave.columns = ['clim']

dset_stations_df = dset_stations_df.T 

# %% and now plot 
for i, row in station_coords.iterrows():
    
    st_name_and_country = f"{row.station_name}, {row.country}"
    
    lon_extracted = float(dset_stations.sel(station_name=st_name_and_country)['lon'])
    lat_extracted = float(dset_stations.sel(station_name=st_name_and_country)['lat'])
    
    sub_df = dset_stations_df.loc[st_name_and_country,]
    
    df_clim = clim_ave.loc[st_name_and_country,]
    
    f = plt.figure(figsize=(12,6))

    ax1 = f.add_axes([0.1,0.25,0.7,0.65])

    sub_df.plot(ax=ax1, color='b', label=None)

    ax1.fill_between(sub_df.index, 0, sub_df.values, color='b', alpha=0.6, label='estimated (MSWEP 2.8.0)')
    
    [l.set_fontsize(12) for l in ax1.xaxis.get_ticklabels()]
    [l.set_fontsize(12) for l in ax1.yaxis.get_ticklabels()]

    ax1.legend(labels=['estimated (MSWEP 2.8.0)'], fontsize=12, loc=2)
    
    ax1.set_title(st_name_and_country)

    ax1.set_xlim(sub_df.index[0], sub_df.index[-1])

    ax1.set_ylim(0, None)

    ax1.set_ylabel("mm", fontsize=12)

    ax1.grid(ls=':')
    
    ax1.set_xlabel('')
    
    sums = [sub_df.sum(), df_clim.values[0]]

    ax2 = f.add_axes([0.8,0.25,0.14,0.65])    
    
    ax2.bar(np.arange(0.5,2.5), sums, width=0.7, color=['b','g'], alpha=0.6, align='center')

    ax2.set_xticks([0.5, 1.5])

    ax2.set_xticklabels(['obs.', 'clim.'], fontsize=12)
    
    ax2.yaxis.tick_right()

    ax2.set_ylabel("mm", fontsize=12)

    ax2.yaxis.set_label_position("right")

    ax2.set_title(f"{np.divide(*np.array(sums)) * 100:4.1f} % of normal", fontdict={'weight':'bold'})

    if lat_extracted < 0:
        ax1.set_title(f"Last {ndays_agg} days to {cycle_time:%Y-%m-%d}, MSWEP 2.8.0 virtual station for {st_name_and_country}\ncoordinates [{lon_extracted:5.2f}E, {lat_extracted*-1:5.2f}S]")
    else: 
        ax1.set_title(f"Last {ndays_agg} days to {cycle_time:%Y-%m-%d}, MSWEP 2.8.0 virtual station for {st_name_and_country}\ncoordinates [{lon_extracted:5.2f}E, {lat_extracted:5.2f}N]")
        
    f.savefig(fig_path.joinpath(f'MSWEP280_Virtual_Station_Time_Series_{utils.sanitize_name(row.station_name)}.png'), dpi=200, bbox_inches='tight', facecolor='w')
    
    plt.close(f)
    
# %% merge the realtime data and the climo 
dset_merge = xr.merge([dset_stations, clim_ave.to_xarray()])

# %% saves to disk 
dset_merge.to_netcdf(opath.joinpath(f'Virtual_Stations_{ndays_agg}days_to_{cycle_time:%Y-%m-%d}.nc'))