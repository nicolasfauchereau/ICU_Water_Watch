import pathlib
from datetime import datetime

from . import utils

def update(ftp_url="data.gloh2o.org", remote_path="niwa_mswep/MSWEP_V280/NRT/Daily", credentials='./MSWEP_credentials.txt', opath="/media/nicolasf/END19101/ICU/data/glo2ho/MSWEP280/NRT/Daily/"):

    import pathlib 
    import ftplib

    opath = pathlib.Path(opath)

    opath.mkdir(parents=True, exist_ok=True)

    with open(credentials, 'r') as f: 
        creds = f.readlines() 

    creds = [c.strip() for c in creds]

    login, password = tuple(creds)

    lfiles_local = list(opath.glob("*.nc"))

    lfiles_local = [f.name for f in lfiles_local]

    lfiles_local.sort()

    ftp = ftplib.FTP_TLS()

    ftp.debugging = 1

    ftp.connect(ftp_url)

    ftp.login(login, password)

    ftp.cwd(remote_path)

    lfiles_remote = ftp.nlst()

    lfiles_remote.sort()

    lfiles_missing = list(set(lfiles_remote) - set(lfiles_local)) 

    lfiles_missing.sort()

    if len(lfiles_missing) >= 1: 

        print(f"preparing to download {len(lfiles_missing)} files missing from the local archive")

        for filename in lfiles_missing: 

            with open(opath.joinpath(filename), 'wb') as f:

                ftp.retrbinary('RETR ' + filename, f.write)

            print(f"retrieved {filename}")
    else: 
    
        pass 

    ftp.close()

def make_dataset(): 

    pass

def set_attrs(dset, ndays=None, last_day=None): 
    """
    set number of days and last day as global attributes 
    in a xarray dataset 

    Parameters
    ----------
    dset : xarray.Dataset
        The xarray Dataset 
    ndays : int, optional
        The number of days, by default None
    last_day : str or datetime, optional
        The last day of the `ndays` period, by default None

    Returns
    -------
    xarray.Dataset
        The xarray Dataset with attributes set
    """
    
    if ndays is not None: 
        
        dset.attrs['ndays'] = ndays 
        
    if last_day is not None: 
        
        if isinstance(last_day, str): 
            dset.attrs['last_day'] = last_day
        elif isinstance(last_day, datetime): 
            dset.attrs['last_day'] = f"{last_day:%Y-%m-%d}"
            
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

    from dask.diagnostics import ProgressBar

    # calculates the accumulation, make sure we keep the attributes 

    dset = dset.sum('time', keep_attrs=True)
    
    with ProgressBar(): 

        dset = dset.compute()
    
    # expand the dimension time to have singleton dimension with last date of the ndays accumulation
    
    dset = dset.expand_dims({'time':[datetime.strptime(dset.attrs['last_day'], "%Y-%m-%d")]})
            
    return dset

def get_rain_days_stats(dset, varname='precipitation', timevar='time', threshold=1, expand_dim=True): 
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
    
    # make sure the correct attributes are set for the latitudes and longitudes 

    dict_lat = dict(units = "degrees_north", long_name = "Latitude")
    dict_lon = dict(units = "degrees_east", long_name = "Longitude")
    
    dset['lat'].attrs.update(dict_lat)
    dset['lon'].attrs.update(dict_lon)

    if opath is None: 
        
        opath = pathlib.Path.cwd() 
        
    else: 
        
        if type(opath) != pathlib.PosixPath: 
            
            opath = pathlib.Path(opath)
            
        if not opath.exists(): 
            
            opath.mkdir(parents=True)
            
    last_date, ndays  = get_attrs(dset)
    
    # build the filename 
    
    filename = f"MSWEP_{kind}_{ndays}ndays_to_{last_date:%Y-%m-%d}.nc"
    
    # saves to disk, using compression level 'complevel' (by default 4)
    
    dset.to_netcdf(opath.joinpath(filename), encoding=dset.encoding.update({'zlib':True, 'complevel':complevel}))
    
    print(f"\nsaving {filename} in {str(opath)}")

def get_virtual_station(dset, lat=None, lon=None, varname='precipitation'): 
    """
    get a time-series from the GPM-IMERG accumulation dataset 

    [extended_summary]

    Parameters
    ----------
    dset : xarray.Dataset
        The input dataset
    lat : [type], optional
        [description], by default None
    lon : [type], optional
        [description], by default None
    varname : str, optional
        [description], by default 'precipitationCal'

    Returns
    -------
    [type]
        [description]
    """
    
    # this to ignore the runtime warning when converting the CFTimeindex to a datetime index 
    import warnings
    warnings.filterwarnings("ignore")
    
    sub = dset.sel(lat=lat, lon=lon, method='nearest')
    
    index = sub.time.to_index().to_datetimeindex()
    
    extracted_lat = float(sub.lat.data)
    extracted_lon = float(sub.lon.data)
    
    dist = utils.haversine((lon, lat), (extracted_lon, extracted_lat))
    
    sub = sub[varname].load()
    
    sub = sub.to_dataframe()[[varname]]
    
    sub.index = index
    
    sub = sub.rename({varname:'observed'}, axis=1)
    
    return sub, (extracted_lon, extracted_lat), dist

def calibrate_SPI(dset, variable='precipitation', dimension='time', return_gamma = False):
    """
    calibrate the SPI over a climatological dataset (typically obtained using `subset_daily_clim`
    with appropriate buffer ...)
    """
    
    import numpy as np 
    import xarray as xr 
    from scipy import stats as st
    
    ds_ma = dset[variable]
    
    ds_In = np.log(ds_ma)
    
    ds_In = ds_In.where(np.isinf(ds_In) == False) #= np.nan  #Change infinity to NaN

    ds_mu = ds_ma.mean(dimension)

    #Overall Mean of Moving Averages
    ds_mu = ds_ma.mean(dimension)

    #Summation of Natural log of moving averages
    ds_sum = ds_In.sum(dimension)

    #Computing essentials for gamma distribution
    n = ds_In.count(dimension)                  #size of data
    A = np.log(ds_mu) - (ds_sum/n)             #Computing A
    alpha = (1/(4*A))*(1+(1+((4*A)/3))**0.5)   #Computing alpha  (a)
    beta = ds_mu/alpha            
    
    if return_gamma: 

        gamma_func = lambda data, a, scale: st.gamma.cdf(data, a=a, scale=scale)

        gamma = xr.apply_ufunc(gamma_func, ds_ma, alpha, beta, dask='allowed')

        return gamma, alpha, beta

    else: 
        
        return alpha, beta
    
def calculate_SPI(dataarray, alpha, beta, name='SPI'): 
    
    import xarray as xr 
    from scipy import stats as st
    
    gamma_func = lambda data, a, scale: st.gamma.cdf(data, a=a, scale=scale)
    
    gamma = xr.apply_ufunc(gamma_func, dataarray, alpha, beta, dask='allowed')
    
    norminv = lambda data: st.norm.ppf(data, loc=0, scale=1)
    
    norm_spi = xr.apply_ufunc(norminv, gamma, dask='allowed')
    
    return norm_spi.to_dataset(name=name)