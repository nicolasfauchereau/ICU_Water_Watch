"""
`utils` module for the ICU Water Watch 
"""


def roll_longitudes(dset, lon_name='lon'): 
    """
    roll the longitudes of a dataset so that it goes from 0 to 360
    instead of -180 to 180

    Parameters
    ----------
    dset : xarray.Dataset
        The input Dataset with the longitudes going from -180 to 180
    lon_name : str, optional
        The name of the longitude dimension, by default 'lon'
        
    Returns
    -------
    
    dset : xarray.Dataset 
        Dataset with rolled longitudes 
    """
    
    dset = dset.assign_coords({lon_name:(dset[lon_name] % 360)}).roll({lon_name:(dset.dims[lon_name] // 2)}, roll_coords=True)
    
    return dset

def interp(dset, interp_factor=4, lon_name='lon', lat_name='lat'): 
    """
    Interpolate (i.e. increase the resolution) of a xarray dataset by `interp_factor`

    Parameters
    ----------
    dset : xarray.Dataset
        the xarray Dataset to interpolate
    interp_factor : int, optional
        the increase in resolution, by default 4
    lon_name : str, optional
        name of the longitude variable, by default 'lon'
    lat_name : str, optional
        name of the latitude variable, by default 'lat'
    
    Return
    ------
    
    dset : the interpolated dataset 
    """
    
    import numpy as np 
    
    target_grid = dset[[lon_name, lat_name]]
    
    target_grid[lon_name] = ((lon_name), np.linspace(target_grid[lon_name].data[0], target_grid[lon_name].data[-1], num=len(target_grid[lon_name])*interp_factor, endpoint=True))
    target_grid[lat_name] = ((lat_name), np.linspace(target_grid[lat_name].data[0], target_grid[lat_name].data[-1], num=len(target_grid[lat_name])*interp_factor, endpoint=True))
    
    dset = dset.interp_like(target_grid)
    
    return dset

def earth_radius(lat):
    from numpy import deg2rad, sin, cos

    lat = deg2rad(lat)
    a = 6378137
    b = 6356752
    r = (
        ((a ** 2 * cos(lat)) ** 2 + (b ** 2 * sin(lat)) ** 2)
        / ((a * cos(lat)) ** 2 + (b * sin(lat)) ** 2)
    ) ** 0.5

    return r

def area_grid(lat, lon, return_dataarray=False):
    """Calculate the area of each grid cell for a user-provided
    grid cell resolution. Area is in square meters, but resolution
    is given in decimal degrees.
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos

    ylat, xlon = meshgrid(lat, lon)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=1))
    dlon = deg2rad(gradient(xlon, axis=0))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    if not return_dataarray:
        return area
    else:
        from xarray import DataArray

        xda = DataArray(
            area.T,
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
            attrs={
                "long_name": "area_per_pixel",
                "description": "area per pixel",
                "units": "m^2",
            },
        )
        return xda
    
def haversine(coords1, coords2):
    """
    returns the distance between 2 points (coordinates in degrees)

    Parameters
    ----------
    coords1: tuple
        (latitude, longitude) of the first point
    coords2: tuple
        (latitude, longitude) of the second point

    Returns
    -------
    float
        distance (in km) between the 2 points
    """

    lat1, lon1 = coords1 
    lat2, lon2 = coords2

    from math import radians, cos, sin, asin, sqrt
    
    R = 6372.8 # this is in km

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))

    return R * c  
    
def make_cluster(n_workers=4, memory_limit=32):
    """
    Creates a dask cluster 
    
    Parameters
    ----------
    
    n_workers : int
        The number of workers, by default 4 
        
    memory_limit : int
        The memory limit for EACH worker, by default 32
    
    """
       
     
    from dask.distributed import Client
    import multiprocessing
    
    ncpu = multiprocessing.cpu_count()
    processes = False
    nworker = n_workers
    threads = ncpu // nworker
    print(
        f"Number of CPUs: {ncpu}, number of threads: {threads}, number of workers: {nworker}, processes: {processes}",
    )
    client = Client(
        processes=processes,
        threads_per_worker=threads,
        n_workers=nworker,
        memory_limit=f"{memory_limit}GB",
    )
    return print(f"dask dashboard available at {client.dashboard_link}") 
    
    
def interp_to_1x(dset): 
    """
    interpolate a GLOBAL dataset to 1deg X 1deg resolution corresponding
    to the GCMs (CDS) coordinate system 
    """
    
    import numpy as np
    import xarray as xr 
    
    d = {}
    d['lat'] = np.arange(-90, 91, 1)
    d['lon'] = np.arange(0, 361, 1)
    d = xr.Dataset(d)
    dset = dset.interp_like(d)
    return dset 

def sanitize_name(name): 
    """
    Sanitizes the name of a country 

    removes all the weird characters 

    Parameters
    ----------
    name : str
        country name 

    Returns
    -------
    str
        The Sanitized country name
    """
    name = name.replace('/','')
    name = name.replace('  ','_')
    name = name.replace('&','')
    name=name.replace(' ','_').replace(':','_')
    return name

def detrend_dim(da, dim, deg=1):
    
    import xarray as xr 
    
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

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
