from . import utils, geo

# list of available GCMs here, not that operationally, we are only using the C3S GCMs: 'ECMWF', 'UKMO', 'METEO_FRANCE', 'DWD', 'CMCC', 'NCEP', 'JMA', 'ECCC'

GCMs = ['ECMWF', 'UKMO', 'METEO_FRANCE', 'DWD', 'CMCC', 'NCEP', 'JMA', 'ECCC', 'KMA', 'NASA', 'MSC']

def download(GCM='ECMWF', system=None, varname='t2m', year=None, month=None, leadtimes=[1,2,3,4,5], opath=None, domain=None, file_format='grib', level='surface', max_retry=3): 
    """
    downloads a forecast / hindcast from the CDS for a given GCM, year and month 

    Parameters
    ----------
    GCM : str, optional
        The name of the GCM [in 'ECMWF','UKMO,'METEO_FRANCE','CMCC, 'DWD', 'NCEP'], by default 'ECMWF'
    system : int, optional, by default None 
        The GCM system number, see https://confluence.ecmwf.int/display/CKB/Description+of+the+C3S+seasonal+multi-system
    varname : str, optional
        Variable name in, by default 't2m'
    year : int, optional
        The year, by default None
    month : int, optional
        The month, by default None
    leadtimes : list, optional
        The list of lead times in months (0 being the initial time), by default [1,2,3,4,5] for 1 to 5 months in the future
     opath : string or pathlib.Path, optional
        The path were to save the forecast files, by default None
    domain : list, optional
        The geographical domain. WARNING !: must be [latN, lonW, latS, lonE], by default None = global
    file_format : string, by default 'grib', but can be 'netcdf' (experimental)
        The file format the file will be downloaded into
    level : str or int, default 'surface'
        The level, either 'surface' (a string) or an int in [950, 500, 200] or appropriate level
    max_retry : int, optional
        The maximum number of retry for the download, by default 3
    """
    
    # imports -----------------------
    import pathlib
    from datetime import datetime
    import cdsapi
    from collections import OrderedDict
    # ------------------------------- 
    
    # defines a dictionnary that maps variable name 

    dvar = OrderedDict()

    if level == 'surface': 

        dvar = OrderedDict()

        # `raw` values, single levels 
        dvar['t2m'] = ['seasonal-monthly-single-levels', '2m_temperature']
        dvar['tprate'] = ['seasonal-monthly-single-levels','total_precipitation']
        dvar['sst'] = ['seasonal-monthly-single-levels','sea_surface_temperature']
        dvar['10m uwind'] = ['seasonal-monthly-single-levels','10m_u_component_of_wind']
        dvar['10m vwind'] = ['seasonal-monthly-single-levels','10m_v_component_of_wind']
        dvar['mslp'] = ['seasonal-monthly-single-levels','mean_sea_level_pressure']
        dvar['latent heat flux'] = ['seasonal-monthly-single-levels','surface_latent_heat_flux']
        dvar['solar radiation'] = ['seasonal-monthly-single-levels','surface_solar_radiation']

        # anomalies, single levels
        dvar['t2m anomaly'] = ['seasonal-postprocessed-single-levels', '2m_temperature_anomaly']
        dvar['tprate anomaly'] = ['seasonal-postprocessed-single-levels', 'total_precipitation_anomalous_rate_of_accumulation']
        dvar['sst anomaly'] = ['seasonal-postprocessed-single-levels', 'sea_surface_temperature_anomaly']
        dvar['10m uwind anomaly'] = ['seasonal-postprocessed-single-levels','10m_u_component_of_wind_anomaly']
        dvar['10m vwind anomaly'] = ['seasonal-postprocessed-single-levels','10m_v_component_of_wind_anomaly']
        dvar['mslp anomaly'] = ['seasonal-postprocessed-single-levels','mean_sea_level_pressure_anomaly']
        dvar['latent heat flux anomaly'] = ['seasonal-postprocessed-single-levels','surface_latent_heat_flux_anomalous_rate_of_accumulation']
        dvar['solar radiation anomaly'] = ['seasonal-postprocessed-single-levels', 'surface_solar_radiation_anomalous_rate_of_accumulation']

    else:
        
        dvar = OrderedDict()
        
        # `raw` values, pressure levels 
        dvar['geopotential'] = ['seasonal-monthly-pressure-levels', 'geopotential']
        dvar['temperature'] = ['seasonal-monthly-pressure-levels', 'temperature']
        dvar['uwind'] = ['seasonal-monthly-pressure-levels', 'u_component_of_wind']
        dvar['vwind'] = ['seasonal-monthly-pressure-levels', 'v_component_of_wind']
        dvar['specific humidity'] = ['seasonal-monthly-pressure-levels', 'specific_humidity']
        
        # anomalies, pressure levels 
        dvar['geopotential anomaly'] = ['seasonal-postprocessed-pressure-levels', 'geopotential']
        dvar['temperature anomaly'] = ['seasonal-postprocessed-pressure-levels', 'temperature']
        dvar['uwind anomaly'] = ['seasonal-postprocessed-pressure-levels', 'u_component_of_wind']
        dvar['vwind anomaly'] = ['seasonal-postprocessed-pressure-levels', 'v_component_of_wind']
        dvar['specific humidity anomaly'] = ['seasonal-postprocessed-pressure-levels', 'specific_humidity']


    # if year and month are not passed, take the current year and month 
    if year is None and month is None: 
        
        year = datetime.utcnow().year

        month = datetime.utcnow().month
        
    if domain is not None: 
        
        # we assume it is passed as [lon_min, lon_max, lat_min, lat_max] (see domains.py)
        # so we need to convert to the order accepted by the cdsapi 
        
        if domain[1] > 180: 
            
            domain[1] = -(360-domain[1])
        
        domain = [domain[-1], domain[0], domain[-2], domain[1]]
    
    else: 
        
        # we assume global
        
        domain = [90, 180, -90, -180]
    
    # if the output path is not defined, will create a folder
    # corresponding to the GCM in the CURRENT directory 
    
    if opath is None: 
        
        opath = pathlib.Path.cwd()
        
        opath = opath.joinpath(GCM)

    else:

    # if opath is defined, but is a string, we cast into pathlib.Path
        
        if not(isinstance(opath, pathlib.Path)): 
            
            opath = pathlib.Path(opath)
    
    # creates the path if not existing on disk
    
    if not(opath.exists()):
        
        opath.mkdir(parents=True)

    # build the filename for output 

    if level == 'surface': 

        # if surface level

        fname_out = opath.joinpath(f"ensemble_seas_forecasts_{varname.replace(' ','_')}_from_{year}_{str(month).zfill(2)}_{GCM}.{file_format}")

        # if the filename has already been downloaded, skip and return the path 
        
        if fname_out.exists():
            
            print(f"\n{str(fname_out)} exists already on disk, skipping download and returning path\n") 

            return fname_out

        # else, connect to the CDS and retrieve file ...
        
        else: 
            
            if system is not None: 
            
                dict_retrieve = {
                            'format':file_format,
                            'originating_centre':GCM.lower(),
                            'system': str(system), 
                            'variable':dvar[varname][1],
                            'product_type':'monthly_mean',
                            'year':str(year),
                            'month':str(month).zfill(2),
                            'leadtime_month': list(map(str, [x+1 for x in leadtimes])), 
                            'area': domain
                            }
            else: 
                
                dict_retrieve = {
                            'format':file_format,
                            'originating_centre':GCM.lower(),
                            'variable':dvar[varname][1],
                            'product_type':'monthly_mean',
                            'year':str(year),
                            'month':str(month).zfill(2),
                            'leadtime_month': list(map(str, [x+1 for x in leadtimes])), 
                            'area': domain
                            }
                
            count = 0

            while not(fname_out.exists()) and count < max_retry:
                
                count += 1

                try: 

                    print(f"\nattempting to download {varname} for GCM {GCM}, for level {level}, year {year}, month {month}, in {file_format}, attempt {count} of {max_retry}\n")
                    
                    c = cdsapi.Client()

                    data = c.retrieve(dvar[varname][0], dict_retrieve, fname_out)

                    if fname_out.exists(): 
                    
                        data.delete()

                        print(f"\n{fname_out} downloaded OK\n")

                        return fname_out

                except: 
                    
                    print(f"\nfailure to download or save {str(fname_out)}\n")
        
    else: 
        
        fname_out = opath.joinpath(f"ensemble_seas_forecasts_{varname.replace(' ','_')}_Z{str(level)}_from_{year}_{str(month).zfill(2)}_{GCM}.{file_format}")

        # if the filename has already been downloaded, skip, and return the path 
        
        if fname_out.exists():

            print(f"\n{str(fname_out)} exists already on disk, skipping download and returning path\n") 

            return fname_out

        # else, connect to the CDS and try retrieve file ...
        
        else: 
            
            if system is not None: 
                
                dict_retrieve = {
                        'format':file_format,
                        'originating_centre':GCM.lower(),
                        'system': str(system),
                        'variable':dvar[varname][1],
                        'pressure_level': str(level), 
                        'product_type':'monthly_mean',
                        'year':str(year),
                        'month':str(month).zfill(2),
                        'leadtime_month': list(map(str, [x+1 for x in leadtimes])), 
                        'area': domain
                        }
            else: 
                
                dict_retrieve = {
                        'format':file_format,
                        'originating_centre':GCM.lower(),
                        'variable':dvar[varname][1],
                        'pressure_level': str(level), 
                        'product_type':'monthly_mean',
                        'year':str(year),
                        'month':str(month).zfill(2),
                        'leadtime_month': list(map(str, [x+1 for x in leadtimes])), 
                        'area': domain
                        }
                
            count = 0

            while not(fname_out.exists()) and count < max_retry: 

                count += 1
                        
                try: 
                    
                    print(f"\nattempting to download {varname} for GCM {GCM}, for level {level}, year {year}, month {month}, in {file_format}, attempt {count} of {max_retry}\n")

                    c = cdsapi.Client()

                    data = c.retrieve(dvar[varname][0], dict_retrieve, fname_out)

                    if fname_out.exists():
                    
                        data.delete()

                        print(f"\n{fname_out} downloaded OK\n")

                        return fname_out

                except: 

                    print(f"\nfailure to download or save {str(fname_out)}\n") 

def convert_rainfall(dset, varin='tprate', varout='precip', leadvar='step', timevar='time', dropvar=True):
    """
    converts the rainfall - anomalies or raw data
    in the CDS forecasts to mm per month ... 
    
    - In the CDS GCMs (ECMWF, UKMO, METEO-FRANCE, DWD, CMCC, NCEP) 
    precipitation is normally expressed in meters per second, and is then converted first to mm/day
    
    - In the CLIKS GCMs (KMA GLOSEA5GC2, NASA GEOS-S2S-2.1, MSC CANSIPSv2, MGO MGOAM-2, HMC SL-AV, BoM ACCESS-S1, 
    BCC CSM1.1M, APCC SCOPS) precipitation is normally expressed in mm/day 
    
    """

    import pandas as pd
    import numpy as np
    from datetime import datetime
    from calendar import monthrange
    from dateutil.relativedelta import relativedelta

    # find the unit 
    
    if 'unit' in dset[varin].attrs: 
        unit = dset[varin].attrs['unit']
    if 'units' in  dset[varin].attrs:
        unit = dset[varin].attrs['units']
            
    # if unit is in meters per second, converts to mm/day 
    if unit in ['m s**-1', 'm/s', 'm^s-1']: 
        
        print(f"\nunit is {unit}, converting to mm/day")
        
        dset[varout] = (dset[varin] * 1000 * (24 * 60 * 60))
        
    elif unit in ['mm/day','mm day**-1', 'mm^day-1']: 
        
        print(f"\nunit is {unit}, doing nothing here")
        
        dset[varout] = dset[varin]

    # gets the dates for each month 
    dates = pd.to_datetime(dset[timevar].data).to_pydatetime()
    
    # gets the number of days in each month 
    ndays = []
    for date in dates: 
        ndays.append([monthrange((date + relativedelta(months=i)).year,(date + relativedelta(months=i)).month)[1]  for i in dset['step'].data])

    # adds this variable into the dataset 
    dset['ndays'] = ((timevar, leadvar), np.array(ndays))

    print(f"\nnow converting to mm/month, converted precipitation will be held in var = {varout}")

    # multiply to get the values in mm / month 
    dset[varout] = dset[varout] * dset['ndays']

    if dropvar:
        dset = dset.drop(varin)
        
    dset = dset.drop('ndays')
    
    dset[varout].attrs = {'units':'mm/month'}

    return dset

def preprocess_GCM(dset, domain=[120, 245, -55, 30]): 
    
    import numpy as np
    import pandas as pd
    import xarray as xr 
    from dateutil.relativedelta import relativedelta
    
    dict_rename = {}
    dict_rename['latitude'] = 'lat'
    dict_rename['longitude'] = 'lon'
    dict_rename['time'] = 'step'
    dict_rename['number'] = 'member'
    
    steps = list(range(1, 6))
    
    # rename the variables 
    
    dset = dset.rename(dict_rename)
    
    # sort by latitude (if going from N to S)
    
    if dset.lat.data[0] > dset.lat.data[-1]: 
    
        dset = dset.sortby('lat')
        
    # roll longitudes (if the first longitude is negative)
    
    if any (dset['lon'] < 0): 
        
        dset = utils.roll_longitudes(dset)
    
    # selects the domain if a list [lonmin, lonmax, latmin, latmax] is passed
    
    if domain is not None: 
        
        dset = dset.sel({'lon':slice(*domain[:2]), 'lat':slice(*domain[2:])})
    
    # if not on a one degree grid we interpolate (e.g. the JMA comes on a 2.5 * 2.5 deg grid originally)
    
    if (all(np.diff(dset['lon']) != 1)) or (all(np.diff(dset['lon']) != 1)): 
        
        target_grid = {}
        target_grid['lon'] = np.arange(dset['lon'].data[0], dset['lon'].data[-1] + 1., 1.)
        target_grid['lat'] = np.arange(dset['lat'].data[0], dset['lat'].data[-1] + 1., 1.)
        target_grid = xr.Dataset(target_grid)
        
        dset = dset.interp_like(target_grid)
    
    # get the init date, we assume here that the first step is leadtime 1 (NOT, I repeat *NOT* 0)
    
    init_date = pd.to_datetime(dset.step.data[0]) - relativedelta(months=1)
    
    # NOW replace the step from datetime to integer leadtime in month
    
    dset['step'] = (('step'), steps)
    
    # add an extra time dimension, containing the initialisation time 
    
    dset = dset.expand_dims(dim={'time':[init_date]}, axis=0) 
    
    # selects always the most recent system if present
    
    if ('system' in dset.dims):
        
        dset = dset.isel(system=-1, drop=True)
        
    return dset 

def calc_climatological_percentiles(dset, percentiles=None, dims=['member','time']):
    """
    calculates the climatological percentiles, over dimensions 
    ['member','time'] from a CDS hindcast dataset 

    Parameters
    ----------
    dset : [type]
        [description]
    percentiles : [type], optional
        [description], by default None
    dims : list, optional
        [description], by default ['member','time']

    Returns
    -------
    xarray.Dataset
        climatological quantiles
        
    Usage
    -----
    
    Usually called inside the `map` method 
    
    e.g. with a xarray.Dataset named `clim`
    
    >> clim = clim.chunk({'time':-1, 'member':-1, 'step':1, 'lat':1, 'lon':1})
    
    >> terciles_climatology = clim.groupby(clim.time.dt.month).map(calc_percentiles, **{'percentiles':[0.3333, 0.6666]})

    >> with ProgressBar():
    >>    terciles_climatology = terciles_climatology.compute()

    """
    
    import numpy as np
    
    if percentiles is None: 
        
        percentiles = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    return dset.quantile(percentiles, dim=dims) 


def get_percentile_bounds(dset, name='quantile'): 
    """
    get the percentile bounds (inserting 0 and 1 at the beginning and the end respectively)
    from a percentile climatology file 

    Parameters
    ----------
    dset : xarray.Dataset 
        The monthly or seasonal percentile dataset, typically tercile, decile, and percentiles 
        0.02, then 0.05 to 0.95 
    name : str, optional
        The name of the variable containing the initial quantiles bounds, by default 'quantile'

    Returns
    -------
    numpy.ndarray   
        The numpy array with the levels (typically, for plotting)
    """
    import numpy as np
    
    percentiles = dset[name].data
    
    percentiles = np.insert(percentiles, 0, 0)
    
    percentiles = np.insert(percentiles, len(percentiles), 1) 
    
    categories =  np.arange(len(percentiles) - 1) + 1
    
    return percentiles, categories 

def get_GCM_category_digitize(dset, clim_quantiles, varname='precip', dim='quantile'): 
    """
    calculates the category of an observation given quantiles

    Parameters
    ----------
    dset : xarray.Dataset 
        The input dataset, typically the GCM forecast
    clim : xarray.Dataset 
        The climatology (needs to vary along a 'dim' dimension as well)
    varname : str, optional
        The name of the variable (needs to be the same in both the input and 
        climatological dataset), by default 'precip'
    dim : str, optional
        The name of the variable describing the quantile in the `clim` dataset, by default 'quantile'

    Returns
    -------
    xarray.Dataset
        The resulting dataset (containing the category)
    """  
    import numpy as np 
    import xarray as xr
    try: 
        import dask 
    except ImportError("dask is not available ..."):
        pass 
    
    def _digitize(x, bins):
        return np.digitize(x.ravel(), bins.ravel())
    
    categories = xr.apply_ufunc(_digitize, dset[varname], clim_quantiles[varname], input_core_dims=[[], [dim]],
                vectorize=True, dask='parallelized')
    
    # add one so that the categories go from 1 to N_categories + 1 
    categories += 1
    
    return categories.to_dataset()

def calculate_quantiles_probabilities(dset, ncategories=3): 
    """
    calculate the probability for each quantile category
    
    Parameters
    ----------
    dset : xarray.Dataset
        The dataset containing the calculated quantile categories (tercile or decile)
        for each member in a GCM ensemble, see function get_GCM_category_digitize 
    ncategories : int, optional
        The number of categories, must be either 3 (for terciles), 10 (deciles) or 21 
        (for percentiles 0.02, then 0.05 to 0.95 with 0.05 step), by default 3

    Returns
    -------
    xarray.Dataset
        The Dataset with the probabilities ([0 - 100]) for 
        each class 

    Raises
    ------
    ValueError
        If the number of categories is not in [3, 10, 21]
    """
    
        
    import xarray as xr 
    import numpy as np 
    
    if not(ncategories in [3, 10, 21]):
        raise ValueError(f"given number of categories {ncategories} is not in [3,10,21]")
        return None
    else:
        category_proportion = []
        # insert tqdm progress bar here
        for category in range(1, ncategories + 1):
            t = dset.where(dset==category).count(dim='member') / len(dset['member']) * 100
            t = t.clip(0, 100)
            t.compute()
            category_proportion.append(t.load()) 
        if ncategories == 3: 
            ds_category_proportion = xr.concat(category_proportion, dim='tercile')
            ds_category_proportion['tercile'] = (('tercile'), np.arange(3) + 1)
        elif ncategories == 10: 
            ds_category_proportion = xr.concat(category_proportion, dim='decile')
            ds_category_proportion['decile'] = (('decile'), np.arange(10) + 1)
        elif ncategories == 21: 
            ds_category_proportion = xr.concat(category_proportion, dim='percentile')
            ds_category_proportion['percentile'] = (('percentile'), np.arange(21) + 1)
               
        return ds_category_proportion.clip(0,100)

def get_GCM_available_period(dpath='/media/nicolasf/END19101/ICU/data/CDS/', GCM='ECMWF', varname='tprate'):
    """
    returns what files are available for what GCM

    [extended_summary]

    Parameters
    ----------
    dpath : str, optional
        [description], by default '/media/nicolasf/END19101/ICU/data/CDS/'
    GCM : str, optional
        [description], by default 'ECMWF'
    varname : str, optional
        [description], by default 'tprate'

    Raises
    ------
    ValueError
        If the GCM is not defined in the list of GCMs
    """
    
    import pathlib
    import xarray as xr 
    
    if not(GCM in GCMs):
        
        raise ValueError(f"{GCM} is not in {','.join(GCMs)}")
    
    dpath_gcm = dpath.joinpath(f"{GCM}/{varname.upper()}")
    
    lfiles_gcm = list(dpath_gcm.glob("*.netcdf"))

    lfiles_gcm.sort()
        
    print(f"The number of monthly files for GCM {GCM} is: {len(lfiles_gcm)}\n") 
    print(f"first file available: {str(lfiles_gcm[0])}\nlast file available: {str(lfiles_gcm[-1])}\n") 
 
def get_one_GCM(dpath='/media/nicolasf/END19101/ICU/data/CDS/', GCM='ECMWF', varname='tprate', start_year=1993, end_year=2016, anomalies=True, ensemble_mean=True, climatology=[1993, 2016], mask=None, domain=None, detrend=False, dropsel=None, load=True): 
    """
    """
    import pathlib
    import xarray as xr  
    import numpy as np 
    import geopandas as gpd 
    
    if not(GCM in GCMs):
        
        raise ValueError(f"{GCM} is not in {','.join(GCMs)}")
    
    vdict = {}
    vdict['tprate'] = 'precip'
    vdict['sst'] = 'sst'
    
    if type(dpath) != pathlib.PosixPath: 
        
        dpath = pathlib.Path(dpath)
    
    dpath_gcm = dpath.joinpath(f"{GCM}/{varname.upper()}")
    
    print(f"getting GCM data from {str(dpath_gcm)}")

    lfiles_gcm = list(dpath_gcm.glob("*.netcdf"))

    lfiles_gcm.sort()
    
    if GCM == 'METEO_FRANCE': 
        lfiles_gcm = [x for x in lfiles_gcm if int(x.name.split('_')[-4]) >= start_year and int(x.name.split('_')[-4]) <= end_year]
    else:
        lfiles_gcm = [x for x in lfiles_gcm if int(x.name.split('_')[-3]) >= start_year and int(x.name.split('_')[-3]) <= end_year]
        
    print(f"reading {len(lfiles_gcm)} files\n") 
    print(f"first: {str(lfiles_gcm[0])}\nlast: {str(lfiles_gcm[-1])}") 
 
    dset_gcm = xr.open_mfdataset(lfiles_gcm, preprocess=preprocess_GCM, parallel=True, engine='netcdf4')
    
    # roll the longitudes if necessary 
    
    if dset_gcm['lon'][0] < 0: # test if the first longitude is negative 
        
        dset_gcm = utils.roll_longitudes(dset_gcm)
    
    # here dropsel 
    
    if dropsel is not None: 
        
        dset_gcm = dset_gcm.drop_sel(dropsel)
    
    if domain is not None: 
        
        dset_gcm =  dset_gcm.sel(lon=slice(*domain[:2]), lat=slice(*domain[2:]))
    
    if ensemble_mean: 
        
        # need to keep the attributes, or else the rainfall units somehow disappear ... 
        # hoping its not causing some issues down the line
        dset_gcm = dset_gcm.mean('member', keep_attrs=True)
        
    # if we are reading the precip forecasts (variable 'tprate') we convert from m.s-1 to mm/month 
    if varname == 'tprate': 
        
        dset_gcm = convert_rainfall(dset_gcm, varin='tprate', varout='precip', leadvar='step', timevar='time', dropvar=True)

    # if we are reading the sst forecasts (and the units is Kelvin) we convert to celsius
    if (varname == 'sst') and (dset_gcm['sst'].attrs['units'] == 'K'): 
        
        dset_gcm['sst'] = dset_gcm['sst'] - 273.15
        dset_gcm['sst'].attrs['units'] = 'C'
        
    # if detrend is set to True, we detrend (assuming over the 'time') dimension
    if detrend: 
        
        dset_gcm[vdict[varname]] = detrend_dim(dset_gcm[vdict[varname]], 'time')

    # is we passed a GeoDataFrame as a mask, we use it to mask the data 
    if (mask is not None) and (type(mask) == gpd.geodataframe.GeoDataFrame): 
    
        dset_gcm = geo.make_mask_from_gpd(dset_gcm, mask, buffer=0.)
        
        dset_gcm = dset_gcm[[vdict[varname]]] * dset_gcm['mask'] 
        
    if anomalies: 
        
        if (climatology[0] == start_year) and (climatology[1] == end_year): 
            
            clim_gcm = dset_gcm.groupby(dset_gcm.time.dt.month).mean('time')
    
            anomalies_gcm = dset_gcm.groupby(dset_gcm.time.dt.month) - clim_gcm
        
        else: 
            
            clim_gcm = dset_gcm.sel(time=slice(str(climatology[0], str(climatology[1]))))
            
            clim_gcm = clim_gcm.groupby(clim_gcm.time.dt.month).mean('time')
            
            anomalies_gcm = dset_gcm.groupby(dset_gcm.time.dt.month) - clim_gcm
            
            if load: 
                
                anomalies_gcm = anomalies_gcm.load()
            
        return anomalies_gcm
    
    else: 
        
        if load: 
            
            dset_gcm = dset_gcm.load()
        
        return dset_gcm
    
def get_GCMs(dpath='/media/nicolasf/END19101/ICU/data/CDS/', GCM='ECMWF', varname='tprate', start_year=1993, end_year=2016, anomalies=True, ensemble_mean=True, climatology=[1993, 2016], mask=None, domain=None, detrend=False, dropsel=None, load=True): 
    """
    """
    import pathlib
    import xarray as xr  
    import numpy as np 
    import geopandas as gpd 
            
    vdict = {}
    vdict['tprate'] = 'precip'
    vdict['sst'] = 'sst'
    
    if (type(GCM) == str) and (GCM != 'MME'): 
        
        print(f"reading {GCM}\n")
        
        dset_gcm = get_one_GCM(dpath=dpath, GCM=GCM, varname=varname, start_year=start_year, end_year=end_year, anomalies=anomalies, climatology=climatology, mask=mask, domain=domain, detrend=detrend, dropsel=dropsel, load=load)
    
    elif (type(GCM) == str) and (GCM == 'MME'): 
        
        list_GCMs = GCMs # list of GCMs is defined at the top level of this module 
        
        dset_l = []
        
        for one_GCM in list_GCMs: 
            
            print('---------------------------------------------------')
            print(f"reading {one_GCM}\n")
            
            dset_gcm = get_one_GCM(dpath=dpath, GCM=one_GCM, varname=varname, start_year=start_year, end_year=end_year, anomalies=anomalies, ensemble_mean=ensemble_mean, climatology=climatology, mask=mask, domain=domain, detrend=detrend,  dropsel=dropsel, load=load)
 
            dset_gcm = dset_gcm.expand_dims({'GCM':[one_GCM]})
            
            dset_l.append(dset_gcm)
            
        dset_gcm = xr.concat(dset_l, dim='GCM')

    elif type(GCM) == list: 
        
        list_GCMs = GCM
        
        dset_l = []
        
        for one_GCM in list_GCMs: 
            
            print('---------------------------------------------------')
            print(f"reading {one_GCM}\n")
            
            dset_gcm = get_one_GCM(dpath=dpath, GCM=one_GCM, varname=varname, start_year=start_year, end_year=end_year, anomalies=anomalies, climatology=climatology, mask=mask, domain=domain, detrend=detrend,  dropsel=dropsel, load=load)
 
            dset_gcm = dset_gcm.expand_dims({'GCM':[one_GCM]})
            
            dset_l.append(dset_gcm)
            
        dset_gcm = xr.concat(dset_l, dim='GCM')    
        
    else: 
        
        raise ValueError(f"GCM should be either a GCM in {', '.join(GCMs)}, MME for the MME, or a list of GCMs in {', '.join(GCMs)}")
        
    return dset_gcm

def calc_percentiles(dset, percentiles=None, dims=['member','time']):

    """
    calculates the climatological percentiles, over dimensions 
    ['member','time'] from a CDS hindcast dataset 
    
    Arguments
    ---------
    
    Returns
    -------
    """
    
    import numpy as np
    
    if percentiles is None: 
        # note: enter *manually* as a list because of rounding issues when using np.insert / np.arange 
        percentiles = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    return dset.quantile(percentiles, dim=dims) 

def CDS_variables(): 
    
    """
    This function just lists the available variables

    [extended_summary]
    """
    
    from collections import OrderedDict
    from pprint import pprint
    
    dvar = OrderedDict()

    # `raw` values, single levels 
    dvar['t2m'] = ['seasonal-monthly-single-levels', '2m_temperature']
    dvar['tprate'] = ['seasonal-monthly-single-levels','total_precipitation']
    dvar['sst'] = ['seasonal-monthly-single-levels','sea_surface_temperature']
    dvar['10m uwind'] = ['seasonal-monthly-single-levels','10m_u_component_of_wind']
    dvar['10m vwind'] = ['seasonal-monthly-single-levels','10m_v_component_of_wind']
    dvar['mslp'] = ['seasonal-monthly-single-levels','mean_sea_level_pressure']
    dvar['latent heat flux'] = ['seasonal-monthly-single-levels','surface_latent_heat_flux']
    dvar['solar radiation'] = ['seasonal-monthly-single-levels','surface_solar_radiation']

    # anomalies, single levels
    dvar['t2m anomaly'] = ['seasonal-postprocessed-single-levels', '2m_temperature_anomaly']
    dvar['tprate anomaly'] = ['seasonal-postprocessed-single-levels', 'total_precipitation_anomalous_rate_of_accumulation']
    dvar['sst anomaly'] = ['seasonal-postprocessed-single-levels', 'sea_surface_temperature_anomaly']
    dvar['10m uwind anomaly'] = ['seasonal-postprocessed-single-levels','10m_u_component_of_wind_anomaly']
    dvar['10m vwind anomaly'] = ['seasonal-postprocessed-single-levels','10m_v_component_of_wind_anomaly']
    dvar['mslp anomaly'] = ['seasonal-postprocessed-single-levels','mean_sea_level_pressure_anomaly']
    dvar['latent heat flux anomaly'] = ['seasonal-postprocessed-single-levels','surface_latent_heat_flux_anomalous_rate_of_accumulation']
    dvar['solar radiation anomaly'] = ['seasonal-postprocessed-single-levels', 'surface_solar_radiation_anomalous_rate_of_accumulation']

    # `raw` values, pressure levels 
    dvar['geopotential'] = ['seasonal-monthly-pressure-levels', 'geopotential']
    dvar['temperature'] = ['seasonal-monthly-pressure-levels', 'temperature']
    dvar['uwind'] = ['seasonal-monthly-pressure-levels', 'u_component_of_wind']
    dvar['vwind'] = ['seasonal-monthly-pressure-levels', 'v_component_of_wind']
    dvar['specific humidity'] = ['seasonal-monthly-pressure-levels', 'specific_humidity']

    # anomalies, pressure levels 
    dvar['geopotential anomaly'] = ['seasonal-postprocessed-pressure-levels', 'geopotential']
    dvar['temperature anomaly'] = ['seasonal-postprocessed-pressure-levels', 'temperature']
    dvar['uwind anomaly'] = ['seasonal-postprocessed-pressure-levels', 'u_component_of_wind']
    dvar['vwind anomaly'] = ['seasonal-postprocessed-pressure-levels', 'v_component_of_wind']
    dvar['specific humidity anomaly'] = ['seasonal-postprocessed-pressure-levels', 'specific_humidity']
    
    pprint(dvar)
