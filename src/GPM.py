"""
The `GPM` module contains all functions related to the *processing* of the GPM-IMERG near realtime satellite derived precipitation for the Southwest Pacific
"""

# ignore user warnings 
import warnings
warnings.simplefilter("ignore", UserWarning)

# import matplotlib 
import matplotlib
# matplotlib.use('Agg') # uncomment to use the non-interactive Agg backend

import sys
import pathlib

from datetime import datetime, timedelta

import pandas as pd 
import xarray as xr

from matplotlib import pyplot as plt

# import the local utils package 

from . import utils

def get_files_list(dpath=None, ndays=None, date=None, lag=None):
    """
    [summary]

    [extended_summary]

    Parameters
    ----------
    dpath : [type], optional
        [description], by default None
    ndays : int, optional
        [description], by default 30
    date : [type], optional
        [description], by default None
    lag : int, optional
        [description], by default 1

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    """
    
    # checks paths 
    if dpath is None: 
        
        dpath = pathlib.Path.cwd().parents[2].joinpath('data/GPM_IMERG/daily') 
    
    else:
        
        if type(dpath) != pathlib.PosixPath: 
            
            dpath = pathlib.Path(dpath)
            
        if not dpath.exists():
             
            raise ValueError(f"The path {str(dpath)} does not exist")
        
    if date is None: 
        
        date =  datetime.utcnow() - timedelta(days=lag)
    
    lfiles = []
    
    for d in pd.date_range(end=date, start=date - timedelta(days=ndays - 1)):
        
        if dpath.joinpath(f"GPM_IMERG_daily.v06.{d:%Y.%m.%d}.nc").exists(): 
            
            lfiles.append(dpath.joinpath(f"GPM_IMERG_daily.v06.{d:%Y.%m.%d}.nc"))
        
        else:
            
            raise ValueError(f"GPM_IMERG_daily.v06.{d:%Y.%m.%d}.nc is missing")
    
    if len(lfiles) != ndays: 
        
        print(f"!!! warning, only {len(lfiles)} days will be used to calculate the rainfall accumulation, instead of the intended {ndays}")
        
    return lfiles


def make_dataset(lfiles=None, dpath=None, varname='precipitationCal', ndays=None, check_lag=True): 
    
    if lfiles is None: 
        
        lfiles = get_files_list(dpath, ndays=ndays)

    dset = xr.open_mfdataset(lfiles, concat_dim='time', combine='by_coords', parallel=True)[[varname]]
    
    # get the last date in the dataset 
    
    last_date = dset.time.to_series().index[-1]
    
    last_date = datetime(last_date.year, last_date.month, last_date.day)
    
    # checks that the lag to realtime does not exceed 2 
    
    if check_lag: 
        
        if (datetime.utcnow() - last_date).days > 2: 
            
            print(f"something is wrong, the last date in the dataset is {last_date:%Y-%m-%d}, the expected date should be not earlier than {datetime.utcnow() - timedelta(days=2):%Y-%m-%d}")
    
    ndays_in_dset = len(dset.time) 

    if ndays_in_dset != ndays: 
        
        print(f"something is wrong with the number of time-steps, expected {ndays}, got {ndays_in_dset}")    
    
    # adds the number of days and the last date as *attributes* of the dataset 
    
    dset.attrs['ndays'] = ndays_in_dset
    dset.attrs['last_day'] = f"{last_date:%Y-%m-%d}"
    
    return dset

def get_attrs(dset): 
    """
    return (in order) the last day and the number of days
    in a (GPM-IMERG) dataset
    
    e.g.:
    
    last_day, ndays = get_attrs(dset)

    Parameters
    ----------
    dset : xarray dataset with global attributes 'last_day' and 'ndays'

    Returns
    -------
    tuple
        last_day, ndays
    """
    
    last_day = datetime.strptime(dset.attrs['last_day'], '%Y-%m-%d')
        
    ndays = dset.attrs['ndays']

    return last_day, ndays


def calculate_realtime_accumulation(dset): 
    """
    """

    # calculates the accumulation, make sure we keep the attributes 

    dset = dset.sum('time', keep_attrs=True)
    
    dset = dset.compute()
    
    # expand the dimension time to have singleton dimension with last date of the ndays accumulation
    
    dset = dset.expand_dims({'time':[datetime.strptime(dset.attrs['last_day'], "%Y-%m-%d")]})
            
    return dset

def convert_rainfall_OBS(dset, varin='precipitationCal', varout='precip', timevar='time'): 
    """
    converts the rainfall - anomalies or raw data - originally in mm/day
    in the GPM-IMERG dataset to mm per month ... 
    """
    
    import pandas as pd 
    import numpy as np 
    from datetime import datetime
    from calendar import monthrange
    from dateutil.relativedelta import relativedelta
    
    # cheks that the variable is expressed in mm/day 
    
    if not('units' in dset[varin].attrs and dset[varin].attrs['units'] in ['mm','mm/day']): 
        print(f"Warning, the variable {varin} has no units attributes or is not expressed in mm or mm/day")
        return None
    else: 
    
        # coonvert cdftime to datetimeindex 
        
        index = dset.indexes[timevar]
        
        if type(index) != pd.core.indexes.datetimes.DatetimeIndex: 
            
            index = index.to_datetimeindex()
    
        dset[timevar] = index
    
        # gets the number of days in each month 
        ndays = [monthrange(x.year, x.month)[1] for x in index]
        
        # adds this variable into the dataset 
        dset['ndays'] = ((timevar), np.array(ndays))
        
        # multiply to get the anomalies in mm / month 
        dset['var'] = dset[varin] * dset['ndays'] 

        # rename 
        dset = dset.drop(varin)
        
        dset = dset.rename({'var':varout})
        
        return dset 
    
def get_climatology(dpath=None, ndays=None, date=None, window_clim=2, lag=None, clim=[2001, 2020]): 
    """
    [summary]

    [extended_summary]

    Parameters
    ----------
    dpath : [type], optional
        [description], by default None
    ndays : int, optional
        [description], by default 30
    date : [type], optional
        [description], by default None
    window_clim : int, optional
        [description], by default 2
    lag : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    
    if dpath is None: 
        
        dpath = pathlib.Path.cwd().parents[1].joinpath('data/GPM_IMERG/daily') 
        
    else:
        
        if type(dpath) != pathlib.PosixPath: 
            
            dpath = pathlib.Path(dpath)
        
        if not dpath.exists(): 
            
            raise ValueError(f"The path {str(dpath)} does not exist")
        
    if date is None: 
        
        date =  datetime.utcnow() - timedelta(days=lag)
    
    clim_file = dpath.joinpath(f'GPM_IMERG_daily.v06.{clim[0]}.{clim[1]}_precipitationCal_{ndays}d_runsum.nc')
    
    dset_clim = xr.open_dataset(clim_file)
    
    dates_clim = [date + timedelta(days=shift) for shift in list(range(-window_clim, window_clim+1))]
    
    time_clim = pd.to_datetime(dset_clim.time.data)

    time_clim = [time_clim[(time_clim.month == d.month) & (time_clim.day == d.day)] for d in dates_clim]

    dset_clim_ref = []
    
    for t in time_clim: 
        
        dset_clim_ref.append(dset_clim.sel(time=t))

    dset_clim_ref = xr.concat(dset_clim_ref, dim='time')
    
    dset_clim.close()
    
    return dset_clim_ref


def calc_anoms_and_pctscores(dset, dset_clim):
    """
    [summary]

    [extended_summary]

    Parameters
    ----------
    dset : [type]
        [description]
    dset_clim : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    
    # anomalies in mm 

    anoms = dset - dset_clim.mean('time')

    # percentage of score, compared to the climatological values 
    
    pctscore = calculate_percentileofscore(dset.squeeze(), dset_clim)
    
    pctscore = pctscore.expand_dims({'time':dset.time})

    dset['pctscore'] = pctscore
    
    dset['anoms'] = anoms['precipitationCal']
    
    return dset

def get_rain_days_stats(dset, varname='precipitationCal', timevar='time', threshold=1, expand_dim=True): 
    """
    return the number of days since last rainfall, and the number of day and wet days 
    according to a threshold defining what is a wet day (by default 1 mm/day)

    Parameters
    ----------
    dset : xarray.Dataset
        The xarray dataset containing the daily rainfall over a period of time
    varname : str, optional
        The name of the precipitation variables, by default 'precipitationCal'
    timevar : str, optional
        The name of the time variable, by default 'time'
    threshold : int, optional
        The threshold (in mm/day) for defining what is a 'rain day', by default 1
    expand_dim : bool, optional
        Whether or not to add a bogus singleton time dimension and coordinate 
        in the dataset, by default 'False'
    
    Return
    ------
    dset : xarray.Dataset
        An xarray dataset with new variables: 
            - wet_days : number of wet days
            - dry_days : number of dry days
            - days_since_rain : days since last rain
    
    """
    
    # imports 
    
    from datetime import datetime
    import numpy as np 
    import xarray as xr
    
    # number of days in the dataset from the attributes
    
    ndays = dset.attrs['ndays'] 

    # last day in the dataset from the attributes   
    last_day = datetime.strptime(dset.attrs['last_day'], "%Y-%m-%d")

    # all values below threshold are set to 0     

    dset = dset.where(dset[varname] >= threshold, other=0)
    
    # all values above or equal to threshold are set to 1 
    
    dset = dset.where(dset[varname] == 0, other=1)
    
    # clip (not really necessary ...)
    
    dset = dset.clip(min=0., max=1.)

    # now calculates the number of days since last rainfall 
    
    days_since_last_rain = dset.cumsum(dim=timevar, keep_attrs=True)
    
    days_since_last_rain[timevar] = ((timevar), np.arange(ndays)[::-1])
    
    days_since_last_rain = days_since_last_rain.idxmax(dim=timevar)
    
    # now calculates the number of wet and dry days
    
    dset = dset.sum(timevar)
    
    number_dry_days = (ndays - dset).rename({varname:'dry_days'})
    
    # put all that into a dataset 
    
    dset = dset.rename({varname:'wet_days'})
    
    dset = dset.merge(number_dry_days)
    
    dset = dset.merge(days_since_last_rain.rename({varname:'days_since_rain'}))
    
    # expand the dimension (add a bogus time dimension with the date of the last day)
    
    if expand_dim: 
        
        dset = dset.expand_dims(dim={timevar:[last_day]}, axis=0)
    
    # make sure the attributes are added back 
    
    dset.attrs['ndays'] = ndays
    dset.attrs['last_day'] = f"{last_day:%Y-%m-%d}"
    
    return dset 

def calculate_percentileofscore(dset, clim, varname='precipitationCal', timevar='time'): 
    """
    calculates the percentile of score of a dataset given a climatology

    [extended_summary]

    Parameters
    ----------
    dset : xarray.Dataset 
        The input dataset, typically the real time GPM/IMERG dataset 
    clim : xarray.Dataset 
        The climatology (needs to vary along a 'timevar' dimension as well)
    varname : str, optional
        The name of the variable (needs to be the same in both the input and 
        climatological dataset), by default 'precipitationCal'
    timevar : str, optional
        The name of the variable describing the time, by default 'time'

    Returns
    -------
    xarray.Dataset
        The resulting dataset (containing the percentile of score)
    """
    
    
    from scipy.stats import percentileofscore
    import xarray as xr 
    
    try: 
        import dask 
    except ImportError("dask is not available ..."):
        pass 
    
    def _percentileofscore(x, y):
        return percentileofscore(x.ravel(), y.ravel(), kind='weak')
    
    pctscore = xr.apply_ufunc(_percentileofscore, clim[varname], dset[varname], input_core_dims=[[timevar], []],
                vectorize=True, dask='parallelized')

    return pctscore

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

def get_dates_to_download(dpath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP', lag=1): 
    
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
    
    print(f"Last downloaded file in {str(dpath)} is {last_file.name}\n")
    
    last_date = get_date_from_file(last_file)
    
    last_date = last_date + timedelta(days=1)
    
    today = date.today() 
    
    download_date = today - relativedelta(days=lag)
    
    dates_to_download = pd.date_range(start=last_date, end=download_date, freq='1D')
    
    return dates_to_download

def update(lag=1, opath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP', proxy=None, lon_min=125., lon_max=240., lat_min=-50., lat_max=25., interp=True): 
    
    import os
    import pathlib
    from subprocess import call
    from shutil import which 
    
    import xarray as xr 
    
    curl = which("curl") 
    
    if not type(opath) == pathlib.PosixPath: 

        opath = pathlib.Path(opath)
    
    # first clean the *.nc4 files if any 
    
    for nc4_file in list(opath.glob("*.nc4")): 

        nc4_file.unlink()
        
    # get the dates 
    
    dates_to_download = get_dates_to_download(opath, lag=lag)
        
    # then loop over the dates, and download the files
    
    for date in dates_to_download:

        root_url = f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.06/{date:%Y/%m}"

        fname = f"3B-DAY-L.MS.MRG.3IMERG.{date:%Y%m%d}-S000000-E235959.V06.nc4"

        fname_out = f'GPM_IMERG_daily.v06.{date:%Y.%m.%d}.nc'

        ### ==============================================================================================================
        # build the command
        if proxy:
            cmd = f"{curl} --silent --proxy {proxy} -n -c ~/.urs_cookies -b ~/.urs_cookies -L --url {root_url}/{fname} -o {opath}/{fname}"
        else:
            cmd = f"{curl} --silent -n -c ~/.urs_cookies -b ~/.urs_cookies -L --url {root_url}/{fname} -o {opath}/{fname}"

        print(f"trying to download {fname_out} in {str(opath)}")

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

                dset_in = utils.roll_longitudes(dset_in)

                dset_in = dset_in.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

                dset_in.to_netcdf(opath.joinpath(fname_out),  unlimited_dims='time')
                
                opath.joinpath(fname).unlink()
                                                    
                dset_in.close()

                trmm_grid.close()

            else:

                print(f'\n! file size for input file {fname} is too small, netcdf file {fname_out} is not yet available to download from {root_url}\n')
                
                # cleaning the nc4 files 
                
                for nc4_file in list(opath.glob("*.nc4")): 

                    nc4_file.unlink()
                
                pass

def save(dset, opath=None, kind='accum', complevel=4): 
    """
    saves a dataset containing either:
    
    - the accumulation statistics (rainfall accumulation, anomalies and percentage of score): kind='accum' or 
    - the nb days statistics: dry days, wet days and days since last rain: kind='ndays'

    Parameters
    ----------
    dset : xarray.Dataset 
        The xarray dataset to save to disk 
    opath : string or pathlib.PosixPath, optional
        The path where to save the dataset, by default None
    kind : str, optional
        The kind of dataset, either 'accum' or 'ndays', by default 'accum'
    complevel : int, optional
        The compression level, by default 4
    """
    if opath is None: 
        
        opath = pathlib.Path.cwd() 
        
    else: 
        
        if type(opath) != pathlib.PosixPath: 
            
            opath = pathlib.Path(opath)
            
        if not opath.exists(): 
            
            opath.mkdir(parents=True)
            
    last_date, ndays  = get_attrs(dset)
    
    # build the filename 
    
    filename = f"GPM_IMERG_{kind}_{ndays}ndays_to_{last_date:%Y-%m-%d}.nc"
    
    # saves to disk, using compression level 'complevel' (by default 4)
    
    dset.to_netcdf(opath.joinpath(filename), encoding=dset.encoding.update({'zlib':True, 'complevel':complevel}))
    
    print(f"\nsaving {filename} in {str(opath)}")
    
    
    