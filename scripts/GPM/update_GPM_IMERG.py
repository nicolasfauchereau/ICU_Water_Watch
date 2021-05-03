#!/home/nicolasf/anaconda3/envs/pangeo/bin/python
# -*- coding: utf-8 -*-

import argparse

def make_trmm_grid(): 
    
    import numpy as np 
    import xarray as xr 
    
    lat_values = np.linspace(-59.875, 59.875, num=480, endpoint=True)
    lon_values = np.linspace(-179.875, 179.875, num=1440, endpoint=True)
    
    d = {}
    d['lat'] = (('lat'), lat_values)
    d['lon'] = (('lon'), lon_values)
    d = xr.Dataset(d)
    
    return d      


def get_date_from_file(filename, sep='.',year_index=-4, month_index=-3, day_index=-2):
    
    import pathlib
    from datetime import date
    from dateutil.relativedelta import relativedelta
    
    if not type(filename) == pathlib.PosixPath: 

        filename = pathlib.Path(filename)
     
    # get the filename 
    fname = filename.name 
    
    fname = fname.split('.')
    
    year = fname[year_index]
    month = fname[month_index]
    day = fname[day_index]
    
    d = list(map(int, [year, month, day])) 
    
    d = date(*d)
    
    return d

def get_dates_to_download(dpath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP', realtime_lag=2): 
    
    import pathlib
    from datetime import date
    from dateutil.relativedelta import relativedelta
    
    import numpy as np
    import pandas as pd

    if not type(dpath) == pathlib.PosixPath: 

        dpath = pathlib.Path(dpath)    
        
    lfiles = list(dpath.glob("GPM_IMERG_daily.v06.????.??.??.nc"))
    
    lfiles.sort()

    last_file = lfiles[-1]
    
    last_date = get_date_from_file(last_file)
    
    today = date.today() 
    
    download_date = today - relativedelta(days=realtime_lag)
    
    dates_to_download = pd.date_range(start=last_date, end=download_date, freq='1D')
    
    return dates_to_download

def download_for_dates(dates, opath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP', proxy=None, lon_min=125., lon_max=240., lat_min=-50., lat_max=25., interp=True): 
    
    import os
    import pathlib
    from subprocess import call
    from shutil import which 
    
    import xarray as xr 
    
    curl = which("curl") 
    
    if not type(opath) == pathlib.PosixPath: 

        opath = pathlib.Path(opath)
    
    # first clean the *.nc4 files 
    
    for nc4_file in list(opath.glob("*.nc4")): 

        nc4_file.unlink()
        
    # then loop over the dates, and download the files
    
    for date in dates:

        root_url = f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.06/{date:%Y/%m}"

        fname = f"3B-DAY-L.MS.MRG.3IMERG.{date:%Y%m%d}-S000000-E235959.V06.nc4"

        fname_out = f'GPM_IMERG_daily.v06.{date:%Y.%m.%d}.nc'

        ### ==============================================================================================================
        # build the command
        if proxy:
            cmd = f"{curl} --silent --proxy {proxy} -n -c ~/.urs_cookies -b ~/.urs_cookies -L --url {root_url}/{fname} -o {opath}/{fname}"
        else:
            cmd = f"{curl} --silent -n -c ~/.urs_cookies -b ~/.urs_cookies -L --url {root_url}/{fname} -o {opath}/{fname}"

        print(cmd)

        # execute the command
        r = call(cmd, shell=True)

        if r != 0:

            print("download failed for date {:%Y-%m-%d}".format(date))
            pass

        else:

            stat_info = os.stat(str(opath.joinpath(fname)))

            if stat_info.st_size > 800000:

                dset_in = xr.open_dataset(opath.joinpath(fname))

                dset_in = dset_in[['HQprecipitation','precipitationCal']]

                if interp: 

                    trmm_grid = make_trmm_grid()

                    dset_in = dset_in.interp_like(trmm_grid)

                dset_in = dset_in.transpose('time','lat','lon')

                # roll in the longitudes to go from -180 → 180 to 0 → 360

                dset_in = dset_in.assign_coords(lon=(dset_in.lon % 360)).roll(lon=(dset_in.dims['lon'] // 2), roll_coords=True)

                dset_in = dset_in.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

                dset_in.to_netcdf(opath.joinpath(fname_out),  unlimited_dims='time')
                
                opath.joinpath(fname).unlink()
                                                    
                dset_in.close()

                trmm_grid.close()

            else:

                print(f'\n! file size for {fname} does not match, netcdf file {fname} probably not available from {root_url}\n')
                
                pass

def main(dpath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP', lon_min=125., lon_max=240., lat_min=-50., lat_max=25., interp=True, proxy=None): 
        
    dates = get_dates_to_download(dpath=dpath)
    
    download_for_dates(dates, opath=dpath, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, proxy=proxy)

# ------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dpath', dest='dpath', type=str, default=None, help='the path where to find AND save the netcdf files, REQUIRED')

    parser.add_argument('-p','--proxy', dest='proxy', type=str, default=None, help='the proxy settings (url:port), default is None (no proxy)')

    parser.add_argument('-lonW','--lon_min', dest='lon_min', type=float, default=125., help='westernmost longitude for the domain to extract, default is 125.')

    parser.add_argument('-lonE','--lon_max', dest='lon_max', type=float, default=240., help='eastermost longitude for the domain to extract, default is 240.')

    parser.add_argument('-latS','--lat_min', dest='lat_min', type=float, default=-50., help='southermost latitude for the domain to extract, default is -50.')

    parser.add_argument('-latN','--lat_max', dest='lat_max', type=float, default=25., help='northermost latitude for the domain to extract, default is 25.')
    
    parser.add_argument('-i','--interp', dest='interp', type=int, default=1, help='whether to interpolate on the TRMM grid, default is 1 (True)')

    vargs = vars(parser.parse_args())

    # pop out the arguments

    dpath = vargs['dpath']
    proxy = vargs['proxy']
    lon_min = vargs['lon_min']
    lon_max = vargs['lon_max']
    lat_min = vargs['lat_min']
    lat_max = vargs['lat_max']
    interp = bool(vargs['interp'])
    
    # calls the main function 
    
    main(dpath=dpath, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, interp=interp, proxy=proxy)
