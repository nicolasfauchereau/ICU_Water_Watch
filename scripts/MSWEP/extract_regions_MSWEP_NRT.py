#!/usr/bin/env python
# coding: utf-8

# %%
import pathlib
import json
import argparse
from datetime import datetime
from ICU_Water_Watch import utils, MSWEP, domains
import xarray as xr
from dask.diagnostics import ProgressBar

# %%
parser = argparse.ArgumentParser(
    prog="extract_domains_MSWEP_NRT.py",
    description="""extract the regional domains defined in `regions.json (mapping region name to [lonmin, lonmax, latmin, latmax])`""",
)

parser.add_argument(
    "-u",
    "--update",
    type=int,
    default=1,
    help="""whether to update the NRT datasets first
    \ndefault 1 (True: Update)""",
)

parser.add_argument(
    "-v",
    "--verbose",
    type=int,
    default=1,
    help="""whether to print something when a file exists already on disk
    \ndefault 1 (True: Verbose)""",
)

parser.add_argument(
    "-c",
    "--credentials",
    type=str,
    default="./MSWEP_credentials.txt",
    help="""Text file with login and password for data.gloh2o.org, only needed if `--update 1`
    \ndefault './MSWEP_credentials.txt'""",
)

parser.add_argument(
    "-i",
    "--ipath",
    type=str,
    default="/media/nicolasf/END19101/ICU/data/glo2ho/MSWEP280/NRT/Daily/",
    help="""The path to the `raw` MSWEP Daily NRT files
    \ndefault `/media/nicolasf/END19101/ICU/data/glo2ho/MSWEP280/NRT/Daily/`""",
)

parser.add_argument(
    "-o",
    "--opath",
    type=str,
    default="/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/",
    help="""The path where to save the extracted files, a directory {country} will be created therein
    \ndefault `/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/`""",
)

parser.add_argument(
    "-d",
    "--regions",
    type=str,
    default="./regions.json",
    help="""A JSON file with the mapping between region names and [lonmin, lonmax, latmin, latmax]
    \ndefault './regions.json'""",
)

# %% parse the arguments
args = parser.parse_args()

# %% get the parameters
update = bool(args.update)
verbose = bool(args.verbose)
credentials = args.credentials
ipath = args.ipath
opath = args.opath
regions = args.regions

# %% cast the paths
dpath_MSWEP_NRT = pathlib.Path(ipath)
dpath_MSWEP_subsets = pathlib.Path(opath)

# %% if update, then we update
if update:
    MSWEP.update(credentials=credentials)

# %% list the NRT files
lfiles_NRT = list(dpath_MSWEP_NRT.glob("*.nc"))
lfiles_NRT.sort()

# %% get the dates from the filenames
lfiles_NRT_dates = [
    datetime.strptime(f"{f.name[:4]} {f.name[4:7]}", "%Y %j") for f in lfiles_NRT
]

# %% read the JSON file containing the mapping between domain name and
with open(regions) as f:
    regions = json.load(f)


# %% function definition
def extract_region_from_files_list(lfiles, lfiles_dates, regions, country, opath):
    country_dir = country.replace(": ", "_").replace(" ", "_")

    opath = opath.joinpath(country_dir)

    opath.mkdir(parents=True, exist_ok=True)

    varname = "precipitation"

    encoding = {varname: {"zlib": True, "shuffle": True, "complevel": 1}}

    for i, fname in enumerate(lfiles):
        date_file = lfiles_dates[i]

        if not opath.joinpath(f"MSWEP_Daily_{date_file:%Y-%m-%d}.nc").exists():
            dset = xr.open_dataset(fname)

            dset = utils.roll_longitudes(dset)

            dset = dset.sortby("lat")

            dset = domains.extract_domain(dset, regions[country])

            with ProgressBar():
                dset = dset.compute()

            dset.attrs["filename_origin"] = fname.name

            dset["DOY"] = (("time"), [int(fname.name[-6:-3])])
            print(
                f"{str(opath.joinpath(f'MSWEP_Daily_{date_file:%Y-%m-%d}.nc'))} does not exists ... saving ..."
            )

            dset.to_netcdf(
                opath.joinpath(f"MSWEP_Daily_{date_file:%Y-%m-%d}.nc"),
                encoding=encoding,
                format="NETCDF4",
            )

            dset.close()

        else:
            if verbose:
                print(
                    f"{str(opath.joinpath(f'MSWEP_Daily_{date_file:%Y-%m-%d}.nc'))} exists ... skipping"
                )

    return opath


def main():
    for cname in regions.keys():
        extract_region_from_files_list(
            lfiles_NRT, lfiles_NRT_dates, regions, cname, dpath_MSWEP_subsets
        )


if __name__ == "__main__":
    main()
