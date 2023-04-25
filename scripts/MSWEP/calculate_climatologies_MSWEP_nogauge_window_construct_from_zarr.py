#!/usr/bin/env python
# coding: utf-8

# %% imports
import pathlib
import shutil
import argparse
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client

# %% MSWEP module
from ICU_Water_Watch import MSWEP


# %% parse the command line arguments
parser = argparse.ArgumentParser(
    prog="calculate_climatologies_MSWEP_nogauge_window_construct_zarr.py",
    description="""
                                calculate the MSWEP climatologies (average, quantiles, SPI alpha and beta params) from the buffered DOY ZARR files for\n
                                the 'no-gauge' version of Daily MSWEP, see `extract_daily_MSWEP_runsum_DOY_with_buffer_to_zarr.py` for the processing of the running accumulation and the creation\n
                                of the ZARR files containing the data varying by year and buffer (7 days each side) for each day of year""",
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
    "-s",
    "--doy_start",
    type=int,
    default=1,
    help="""The day of year to start the loop\\ndefault 1""",
)

parser.add_argument(
    "-e",
    "--doy_stop",
    type=int,
    default=365,
    help="""The day of year to stop the loop
    \ndefault 365""",
)

parser.add_argument(
    "-cs",
    "--clim_start",
    type=int,
    default=1993,
    help="""The start year for the climatological period
    \ndefault 1993""",
)

parser.add_argument(
    "-ce",
    "--clim_stop",
    type=int,
    default=2016,
    help="""The end year for the climatological period
    \ndefault 2016""",
)

parser.add_argument(
    "-v",
    "--varname",
    type=str,
    default="precipitation",
    help="""The variable name
    \ndefault 'precipitation'""",
)

parser.add_argument(
    "-i",
    "--ipath_zarr",
    type=str,
    default="/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/SP/climatologies/",
    help="""The path containing the zarr datasets (1 for each DOY)
    \ndefault `/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/SP/climatologies/`""",
)

parser.add_argument(
    "-d",
    "--dask_dir",
    type=str,
    default="./dask_dir",
    help="""The path to the dask folder, default `./dask_dir`""",
)

args = parser.parse_args()

ndays_agg = args.ndays_agg
doy_start = args.doy_start
doy_stop = args.doy_stop
clim_start = int(args.clim_start)
clim_stop = int(args.clim_stop)
varname = args.varname
ipath_zarr = args.ipath_zarr
dask_dir = args.dask_dir

# %% encodings, hard coded
encodings = {varname: {"zlib": True, "shuffle": True, "complevel": 1}}

### ---------------------------------------------------------------------------------------------------------------------------------------------

# %%%
ipath_zarr = pathlib.Path(ipath_zarr).joinpath(f"{ndays_agg}days")
opath_netcdf = ipath_zarr.joinpath(f"{clim_start}_{clim_stop}/netcdf")

opath_netcdf.mkdir(parents=True, exist_ok=True)

# %%
pathlib.Path(dask_dir).mkdir(parents=True, exist_ok=True)

# %% create dask cluster
# client = Client(
#     threads_per_worker=1, n_workers=12, processes=True, local_directory=dask_dir
# )

# %% list the zarr files
ldirs = list(ipath_zarr.glob("*.zarr"))

ldirs.sort()

# %% main loop
for DOY in np.arange(doy_start, doy_stop + 1):

    dset = xr.open_zarr(ldirs[DOY-1])

    print(f"opening {str(ldirs[DOY-1])}")

    # selects the climatological period

    dset = dset.sel(time=slice(str(clim_start), str(clim_stop)))

    with ProgressBar():
        dset = dset.compute()

    dset_average = dset.mean(dim=("time", "buffer"))

    dset_average = dset_average.expand_dims({"DOY": [DOY]})

    dset_average.to_netcdf(
        opath_netcdf.joinpath(
            f"MSWEP_Daily_nogauge_DOY_{DOY:03d}_{ndays_agg}days_runsum_average.nc"
        ),
        encoding=encodings,
        format="NETCDF4",
    )

    dset_average.close()

    dset = dset.stack(instance=("time", "buffer"))

    dset_quantiles = dset[[varname]].quantile(
        q=[
            0.02,
            0.05,
            0.1,
            0.2,
            0.25,
            0.3,
            0.3333,
            0.4,
            0.5,
            0.6,
            0.6666,
            0.7,
            0.75,
            0.8,
            0.9,
        ],
        dim=("instance"),
        skipna=False,
    )

    dset_quantiles = dset_quantiles.expand_dims({"DOY": [DOY]})

    dset_quantiles.to_netcdf(
        opath_netcdf.joinpath(
            f"MSWEP_Daily_nogauge_DOY_{DOY:03d}_{ndays_agg}days_runsum_quantiles.nc"
        ),
        encoding=encodings,
        format="NETCDF4",
    )

    dset_quantiles.close()

    alpha, beta = MSWEP.calibrate_SPI(dset, variable=varname, dimension="instance")

    alpha = alpha.to_dataset(name="alpha").expand_dims({"DOY": [DOY]})

    dset_spi_params = alpha.merge(
        beta.to_dataset(name="beta").expand_dims({"DOY": [DOY]})
    )

    spi_encodings = {}

    for v in dset_spi_params.data_vars:
        spi_encodings[v] = {"zlib": True, "shuffle": True, "complevel": 1}

    dset_spi_params.to_netcdf(
        opath_netcdf.joinpath(
            f"MSWEP_Daily_nogauge_DOY_{DOY:03d}_{ndays_agg}days_runsum_SPI_params.nc"
        ),
        encoding=spi_encodings,
        format="NETCDF4",
    )

    dset_spi_params.close()


# %% remove the dask folder
shutil.rmtree(dask_dir)

# EOF
