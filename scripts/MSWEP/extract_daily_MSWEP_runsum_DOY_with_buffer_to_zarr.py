#!/usr/bin/env python
# coding: utf-8

import pathlib
import argparse
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser(
    prog="extract_daily_MSWEP_runsum_DOY_with_buffer_to_zarr.py",
    description="""construct zarr datasets for each DOY from a netcdf file containing running accumulations""",
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
    help="""The day of year to start the loop
    \ndefault 1""",
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
    "-f",
    "--fname",
    type=str,
    help="""The input filename, e.g. 'merged_1979_2020_chunked_noleap_runsum_{ndays_agg}.nc'
    \nno default""",
)

parser.add_argument(
    "-o",
    "--opath",
    type=str,
    default="/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/SP",
    help="""The directory root where to save the zarr datasets, a 'climatologies/${ndays_agg}days' directory will be created therein
    \ndefault /media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/SP""",
)

parser.add_argument(
    "-w",
    "--window",
    type=int,
    default=15,
    help="""The size of the window in days, should be an odd number, the buffer is defined as {window}//2
    \ndefault is 15 (7 days each side of the target DOY)""",
)

args = parser.parse_args()

ndays_agg = args.ndays_agg
doy_start = args.doy_start
doy_stop = args.doy_stop
fname = pathlib.Path(args.fname)
opath = pathlib.Path(args.opath)
window = args.window

# ----------------------------------------------------------------------------------------------------

opath = opath.joinpath(f"climatologies/{ndays_agg}days")

opath.mkdir(parents=True, exist_ok=True)

dset = xr.open_dataset(fname)

dset = dset.chunk({"time": -1, "lat": 200, "lon": 200})

dset = dset.drop("time_bnds")

dset = dset.rolling({"time": window}, center=True, min_periods=window).construct(
    window_dim="buffer"
)

for DOY in np.arange(doy_start, doy_stop + 1):
    print(f"processing DOY {DOY:03d} ...")

    doy_dset = dset.sel(time=(dset.time.dt.dayofyear == DOY))

    doy_dset.to_zarr(
        opath.joinpath(
            f"MSWEP_nogauge_{ndays_agg}_days_runsum_{window // 2}_days_buffer_DOY_{DOY:03d}.zarr"
        )
    )

    doy_dset.close()

dset.close()
