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


def make_dataset(lfiles=None, dpath=None, ndays=None, check_lag=True): 
    
    if lfiles is None: 
        
        lfiles = get_files_list(dpath, ndays=ndays)

    dset = xr.open_mfdataset(lfiles, concat_dim='time', combine='by_coords', parallel=True)[['precipitationCal']]
    
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
