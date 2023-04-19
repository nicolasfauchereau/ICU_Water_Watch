#!/usr/bin/env python
# coding: utf-8

# %% 
import pathlib
import json 
import argparse
from datetime import datetime
from ICU_Water_Watch import utils, MSWEP
import xarray as xr
from dask.diagnostics import ProgressBar 

# %%
parser = argparse.ArgumentParser(
    prog="extract_domains_MSWEP_NRT.py",
    description="""extract """,
)

parser.add_argument(
    "-u",
    "--update",
    type=int,
    default=1,
    help="""whether to update the NRT datasets first, default 1 (True: update)""",
)

parser.add_argument(
    "-c",
    "--credentials",
    type=str,
    default='./MSWEP_credentials.txt',
    help="""Text file with login and password for data.gloh2o.org, only needed if `--update 1`, default './MSWEP_credentials.txt'""",
)

parser.add_argument(
    "-i",
    "--ipath",
    type=str,
    default='/media/nicolasf/END19101/ICU/data/glo2ho/MSWEP280/NRT/Daily/',
    help="""The path to the `raw` MSWEP Daily NRT files, default """,
)

parser.add_argument(
    "-o",
    "--opath",
    type=str,
    default='/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/',
    help="""The path where to save the extracted files, a directory {country} will be created therein""",
)

parser.add_argument(
    "-d",
    "--domains",
    type=str,
    default='./domains.json',
    help="""A JSON file with the mapping between domain names and [lonmin, lonmax, latmin, latmax], default './domains.json'""",
)

# %% parse the arguments 
args = parser.parse_args()

# %% get the parameters 
update = bool(args.update)
credentials = args.credentials
ipath = args.ipath 
opath = args.opath 
domains = args.domains

# %% cast the paths 
dpath_MSWEP_NRT = pathlib.Path(ipath)
dpath_MSWEP_subsets  = pathlib.Path(opath) 

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
with open(domains) as f: 
    
    domains = json.load(f)

# %% function definition 
def extract_domain_from_files_list(lfiles, lfiles_dates, domains, country, opath): 
    
    country_dir = country.replace(': ','_').replace(' ','_')
    
    opath = opath.joinpath(country_dir)
    
    opath.mkdir(parents=True, exist_ok=True)
    
    varname = 'precipitation'

    encoding={varname:{'zlib':True, 'shuffle':True, 'complevel':1}}
    
    for i, fname in enumerate(lfiles):
    
        date_file = lfiles_dates[i]
        
        if not opath.joinpath(f'MSWEP_Daily_{date_file:%Y-%m-%d}.nc').exists(): 
            
            dset = xr.open_dataset(fname)
    
            dset = utils.roll_longitudes(dset)

            dset = dset.sortby("lat")

            dset = domains.extract_domain(dset, domains[country])
    
            with ProgressBar(): 
        
                dset = dset.compute()
    
            dset.attrs['filename_origin'] = fname.name
    
            dset['DOY'] = (('time'), [int(fname.name[-6:-3])])
        
            print(f"{str(opath.joinpath(f'MSWEP_Daily_{date_file:%Y-%m-%d}.nc'))} does not exists ... saving ...")
        
            dset.to_netcdf(opath.joinpath(f'MSWEP_Daily_{date_file:%Y-%m-%d}.nc'),  encoding=encoding, format='NETCDF4')
            
            dset.close()
        
        else: 
            
            print(f"{str(opath.joinpath(f'MSWEP_Daily_{date_file:%Y-%m-%d}.nc'))} exists ... skipping")
    
    return opath


def main(): 

    for cname in domains.keys(): 

        extract_domain_from_files_list(lfiles_NRT, lfiles_NRT_dates, domains, cname, dpath_MSWEP_subsets)

if __name__ == "__main__":

    main() 