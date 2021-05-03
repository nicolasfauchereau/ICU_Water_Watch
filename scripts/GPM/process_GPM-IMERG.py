#!/home/nicolasf/anaconda3/envs/pangeo/bin/python
# coding: utf-8

# ignore user warnings 
import warnings
warnings.simplefilter("ignore", UserWarning)

# import matplotlib, use the non-interactive Agg backend 
import matplotlib
matplotlib.use('Agg')

import sys
import pathlib
import argparse

from datetime import datetime, timedelta

import pandas as pd 
import xarray as xr

from matplotlib import pyplot as plt

# ### import the local package 

sys.path.append('../../code/')

from hotspot import plot, utils

### define the functions used in the script 

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

def get_dset(lfiles=None, dpath=None, ndays=None, check_lag=True): 
    
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

def calculate_realtime_accumulation(dset): 
    """
    """

    # calculates the accumulation, make sure we keep the attributes 

    dset = dset.sum('time', keep_attrs=True)
    
    dset = dset.compute()
    
    # expand the dimension time to have singleton dimension with last date of the ndays accumulation
    
    dset = dset.expand_dims({'time':[datetime.strptime(dset.attrs['last_day'], "%Y-%m-%d")]})
            
    return dset
    
def get_climatology(dpath=None, ndays=None, date=None, window_clim=2, lag=None): 
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
    
    clim_file = dpath.joinpath(f'GPM_IMERG_daily.v06.2001.2019_precipitationCal_{ndays}d_runsum.nc')
    
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
    
    pctscore = utils.calculate_percentileofscore(dset.squeeze(), dset_clim)
    
    pctscore = pctscore.expand_dims({'time':dset.time})

    dset['pctscore'] = pctscore
    
    dset['anoms'] = anoms['precipitationCal']
    
    return dset

def get_EEZs(dpath_shapes=None): 
    """
    [summary]

    [extended_summary]

    Parameters
    ----------
    dpath_shapes : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    
    if dpath_shapes is None: 
        
        dpath_shapes = pathlib.Path.cwd().parents[2].joinpath('shapefiles')
        
    else: 
        
        if type(dpath_shapes) != pathlib.PosixPath: 
            
            dpath_shapes = pathlib.Path(dpath_shapes)

    if not dpath_shapes.exists(): 
        
        raise ValueError(f"{str(dpath_shapes)} does not exist")
    
    EEZs = utils.read_shapefiles(dpath_shapes.joinpath('EEZs'), filename='ICU_geometries0_360_EEZ.shp')
    
    merged_EEZs = utils.read_shapefiles(dpath_shapes.joinpath('EEZs'), filename='ICU_geometries0_360_EEZ.shp', merge=True, buffer=0.05)
    
    return EEZs, merged_EEZs

def get_coastlines(dpath_shapes=None): 
    """
    [summary]

    [extended_summary]

    Parameters
    ----------
    dpath_shapes : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """

    if dpath_shapes is None: 
        
        dpath_shapes = pathlib.Path.cwd().parents[2].joinpath('shapefiles')
        
    else: 
        
        if type(dpath_shapes) != pathlib.PosixPath: 
            
            dpath_shapes = pathlib.Path(dpath_shapes)

    if not dpath_shapes.exists(): 
        
        raise ValueError(f"{str(dpath_shapes)} does not exist")
    
    coastlines = utils.read_shapefiles(dpath_shapes.joinpath('Coastlines'), filename='ICU_geometries0_360_coastlines.shp')
        
    return coastlines

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
            
    ndays = dset.attrs['ndays']

    last_day = dset.attrs['last_day']
    
    # build the filename 
    
    filename = f"GPM_IMERG_{kind}_{ndays}ndays_to_{last_day}.nc"
    
    # saves to disk, using compression level 'complevel' (by default 4)
    
    dset.to_netcdf(opath.joinpath(filename), encoding=dset.encoding.update({'zlib':True, 'complevel':complevel}))
    
    print(f"\nsaving {filename} in {str(opath)}")
    
def main(dpath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP', ndays=30, lag=1, dpath_shapes=None, opath=None, fpath=None): 
    """
    [summary]

    [extended_summary]

    Parameters
    ----------
    dpath : [type], optional
        [description], by default None
    ndays : int, optional
        [description], by default 30
    dpath_shapes : [type], optional
        [description], by default None
    """
    
    # get the list of files, and checks for any problems such as missing files 
    
    print("\ngetting the list of GPM-IMERG realtime files ...")
    
    lfiles = get_files_list(dpath=dpath, ndays=ndays, date=None, lag=lag)
    
    print(f"\nfirst file is:\n{str(lfiles[0])}\nlast file is:\n{str(lfiles[-1])}")
    
    # return an xarray dataset, with some attributes 
    
    dset = get_dset(lfiles, ndays=ndays)
    
    # calculate the number of dry days, wet days and days since last rain 
    # variables: 'dry_days', 'wet_days', 'days_since_rain' 
    
    print(f"\ncalculating wet, dry days and number of days since last rain, period {ndays} days")
    
    dset_ndays = utils.get_rain_days_stats(dset)
    
    # calculates the rainfall accumulation over the past N days 
    
    print(f"\ncalculating accumulation over {ndays} days")
    
    dset_accum = calculate_realtime_accumulation(dset)
            
    # get the corresponding climatology 
    dset_clim = get_climatology(dpath=dpath, ndays=ndays, date=None, lag=lag)
    
    # add the anomalies and the percentage of score 
    # variables: 'anoms', 'pctscore'
    
    print(f"\ncalculating anomalies and percentile of score for {ndays} accumulation")
    
    dset_accum =  calc_anoms_and_pctscores(dset_accum, dset_clim) 
            
    # get the EEZs (individual Island groups) as well as the 'merged' EEZs for the mask
    EEZs, merged_EEZs = get_EEZs(dpath_shapes=dpath_shapes)
        
    # adds the EEZs (merged) mask to the dataset of rainfall accumulation, anomalies and percentage of score  
    dset_accum = utils.make_mask_from_gpd(dset_accum, merged_EEZs, insert=True, mask_name='EEZs', subset=False)
    
    # adds the EEZs (merged) mask to the dataset containing the number of dry, wet days and days since last rain
    dset_ndays = utils.make_mask_from_gpd(dset_ndays, merged_EEZs, insert=True, mask_name='EEZs', subset=False)
        
    # saves to disk 
    
    # set up path to netcdf files 
    
    if type(opath) != pathlib.PosixPath: 
        
        opath = pathlib.Path(opath)
        
    save(dset_accum, opath=opath, kind='accum')
        
    save(dset_ndays, opath=opath, kind='ndays')
    
    # set up path to the figures 
    
    if type(fpath) != pathlib.PosixPath: 
        
        fpath = pathlib.Path(fpath)
        
    print("\nnow plotting the regional (Pacific) maps")
    # plots the rainfall accumulation 
    
    plot.map_precip_accum(dset_accum, geoms=EEZs, fpath=fpath, close=True)
    
    # plots the rainfall anomalies 
    
    plot.map_precip_anoms(dset_accum, geoms=EEZs, fpath=fpath, close=True)
    
    # plot the deciles
    
    plot.map_decile(dset_accum, geoms=EEZs, fpath=fpath, close=True)
    
    # plots the Pacific EAR Watch 
    
    plot.map_EAR_Watch_Pacific(dset_accum, geoms=EEZs, fpath=fpath, close=True)
    
    # plots the Pacific US Drought Monitor
    
    plot.map_USDM_Pacific(dset_accum, geoms=EEZs, fpath=fpath, close=True)
    
    # plots the number of dry days 
    
    plot.map_dry_days_Pacific(dset_ndays, geoms=EEZs, fpath=fpath, close=True)
    
    # plots the number of days since last rain 
    
    plot.map_days_since_rain_Pacific(dset_ndays, geoms=EEZs, fpath=fpath, close=True)
    
    # now onto the REGIONAL EAR Watch and regional US Drought Monitor
    
    # get the coastlines (to be used for the regional plots)
    
    coastlines = get_coastlines(dpath_shapes=dpath_shapes)
    
    # interpolate at high resolution 
    
    dset_accum_interp = utils.interp(dset_accum)
    
    # now loops over each country and plots 
    
    print("\nnow processing the individual countries / Island groups")
    
    for country_name in coastlines.country_na.values: 
    
        print(f"\n  --> processing {country_name}, EAR Watch then USDM")
    
        coastline = coastlines.query(f"country_na == '{country_name}'")
    
        EEZ = EEZs.query(f"COUNTRYNAM == '{country_name}'")
    
        # subset and add EEZ mask 
        dset_sub_country = utils.make_mask_from_gpd(dset_accum_interp, EEZ, subset=True, mask_name='mask_EEZ')
    
        # add country mask 
        dset_sub_country = utils.make_mask_from_gpd(dset_sub_country, coastline, subset=False, mask_name='mask_coastline')
    
        # plots the regional EAR Watch and US 'Drough Monitor' for each country / Island group, with masked EEZ, then masked coastlines
        for mask_type in ['mask_EEZ', 'mask_coastline']: 
            
            # country EAR Watch 
            
            plot.map_EAR_Watch(dset_sub_country, coastlines, coastline, EEZ, mask_name=mask_type, country_name=country_name, fpath=fpath.joinpath('EAR_Watch'), close=True)    
            
            # country US Drought Monitor
            
            plot.map_USDM(dset_sub_country, coastlines, coastline, EEZ, mask_name=mask_type, country_name=country_name, fpath=fpath.joinpath('USDM'), close=True)
    
    
# -------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dpath', dest='dpath', type=str, default=None, help='the path where to find the GPM-IMERG realtime netcdf files and climatologies, REQUIRED')
    
    parser.add_argument('-n','--ndays', dest='ndays', type=int, default=30, help='the number of days over which to calculate the accumulation and take the climatology')

    parser.add_argument('-l','--lag', dest='lag', type=int, default=1, help='the lag (in days) to realtime, if run in the morning (NZ time) 1 should be OK')

    parser.add_argument('-ds','--dpath_shapes', dest='dpath_shapes', type=str, default=None, help='the path to the `EEZs` and `Coastlines` folder containing the respective shapefiles, REQUIRED')

    parser.add_argument('-o','--opath', dest='opath', type=str, default=None, help='the path where to save the outputs files (netcdf and geotiff), REQUIRED')
    
    parser.add_argument('-f','--fpath', dest='fpath', type=str, default=None, help='the path where to save the figures, REQUIRED')

    vargs = vars(parser.parse_args())

    # pop out the arguments

    dpath = vargs['dpath']
    ndays = vargs['ndays']
    lag = vargs['lag']
    dpath_shapes = vargs['dpath_shapes']
    opath = vargs['opath']
    fpath = vargs['fpath']
    
    # call main
  
    main(dpath=dpath, ndays=ndays, lag=lag, dpath_shapes=dpath_shapes, opath=opath, fpath=fpath)