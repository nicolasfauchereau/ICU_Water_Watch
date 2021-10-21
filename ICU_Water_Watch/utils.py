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
    """
    return the earth radius for a given latitude

    Parameters
    ----------
    lat : float
        The latitude

    Returns
    -------
    Float
        The Earth radius
    """
    
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
    returns the distance between 2 points (given coordinates in degrees)

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
    
    
def interp_to_1x(dset, lon_name='lon', lat_name='lat'): 
    """
    interpolate a dataset to 1deg X 1deg resolution corresponding
    to the GCMs (CDS) coordinate system 
    """
    
    import numpy as np
    import xarray as xr 
    
    if dset[lon_name][0] < 0: 
        dset = roll_longitudes(dset, lon_name=lon_name)

    if dset[lat_name][0] > dset[lat_name][-1]: 
        dset = dset.sort_by(lat_name)
        
    d = {}
    d['lat'] = np.arange(float(dset[lat_name].data[0]), float(dset[lat_name].data[-1]) + 1, 1.)
    d['lon'] = np.arange(float(dset[lon_name].data[0]), float(dset[lon_name].data[-1]) + 1, 1.)
    
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
    name= name.replace(' ','_')
    name = name.replace(':','_')
    name = name.replace('__','_')
    name = name.replace("'","")
    return name

def detrend_dim(da, dim, deg=1, add_average=True):
    
    import xarray as xr 
    
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    
    # return the detrended + average
    
    if add_average: 
    
        return (da - fit) + da.mean(dim=dim)

    else: 
        
        return (da - fit)
        

def pixel2poly(dset, varname=None, lon_name='lon', lat_name='lat'):
    """
    
    converts pixels (grid points) in a dataarray to shapely polygons
    
    modified from: 
    
    https://github.com/TomasBeuzen/python-for-geospatial-analysis/tree/main/chapters/scripts
        
    """
    import itertools
    import numpy as np
    import pandas as pd
    import xarray as xr
    from shapely.geometry import Polygon
    
    x = dset[lon_name].data 
    y = dset[lat_name].data
    
    # note that it only works for 2D data 
    
    z = dset[varname].squeeze().data 
    
    polygons = []
    values = []
    half_res = resolution / 2
    for i, j  in itertools.product(range(len(x)), range(len(y))):
        minx, maxx = x[i] - half_res, x[i] + half_res
        miny, maxy = y[j] - half_res, y[j] + half_res
        polygons.append(Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]))
        if isinstance(z, (int, float)):
            values.append(z)
        else:
            values.append(z[j, i])
    return polygons, values

def _interpolate_NaN(data):
    """

    """
    import numpy as np 
    from scipy import interpolate
    
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    
    # mask invalid values
    array = np.ma.masked_invalid(data)
    
    # get grid
    xx, yy = np.meshgrid(x, y)
    
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    
    newarr = array[~array.mask]
    
    interp = interpolate.NearestNDInterpolator(list(zip(x1, y1)), newarr)
    
    x_out = x
    y_out = y
    
    xx, yy = np.meshgrid(x_out, y_out)
    
    return interp(xx, yy)

def interpolate_NaN_da(dataarray, lon_name='lon', lat_name='lat'): 

    import xarray as xr

    regridded = xr.apply_ufunc(_interpolate_NaN, dataarray,
                           input_core_dims=[[lat_name,lon_name]],
                           output_core_dims=[[lat_name,lon_name]],
                           vectorize=True, dask="allowed")
    
    return regridded

    

def read_config(basepath, fname): 
    """
    read a YAML configuration file 
    containing one document, and return a 
    dictionnay (mapping key --> value)

    see https://pynative.com/python-yaml/

    Parameters
    ----------
    basepath : str
        The basepath 
    fname : str
        The filename

    Returns
    -------
    dictionnary 
        The dictionnary with key (parameter) mapping to value
    """
    
    import pathlib
    import yaml
    from yaml import BaseLoader as Loader
    
    with open(pathlib.Path(basepath).joinpath(fname), 'r') as f:
        mapping = yaml.load(f, Loader=Loader)
    return mapping 

def round_to_100_percent(number_set, digit_after_decimal=2):
    """
    This function take a list of number and return a list of percentage, which represents the portion of each number in sum of all numbers
    Moreover, those percentages are adding up to 100%!!!
    Notice: the algorithm we are using here is 'Largest Remainder'
    The down-side is that the results won't be accurate, but they are never accurate anyway:)
    """
    unround_numbers = [x / float(sum(number_set)) * 100 * 10 ** digit_after_decimal for x in number_set]
    decimal_part_with_index = sorted([(index, unround_numbers[index] % 1) for index in range(len(unround_numbers))], key=lambda y: y[1], reverse=True)
    remainder = 100 * 10 ** digit_after_decimal - sum([int(x) for x in unround_numbers])
    index = 0
    while remainder > 0:
        unround_numbers[decimal_part_with_index[index][0]] += 1
        remainder -= 1
        index = (index + 1) % len(number_set)
    return [int(x) / float(10 ** digit_after_decimal) for x in unround_numbers]