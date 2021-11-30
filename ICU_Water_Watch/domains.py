# imports 

from . import geo 
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs

# dictionnay holding the different geographical domains used 
# in the ICU Water Watch and the SCO 

domains = {}
domains['Tropical_Pacific'] = [140, 360-140, -25, 25]
domains['SW_Pacific'] = [172.5, 190, -22.5, -12]
domains['Fiji'] = [175, 183, -21, -15]
domains['NZ'] = [161, 181, -50, -30] 
domains['Pacific'] = [140, 240, -50, 25]
domains['C3S_download'] = [100, 240, -50, 30]
domains['Water_Watch'] = [120, 240, -35, 25]


def plot_domain(domains):
    """
    small function to plot a dictionnary of domains ([lonmin, lonmax, latmin, latmax])

    Parameters
    ----------
    domains : dictionnary
        key = name
        value = list ([lonmin, lonmax, latmin, latmax])

    Returns
    -------
    matplotlib Figure
    """
    
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(domains.domains))))

    f, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)), figsize=(13,8))

    for k in domains.domains.keys(): 
    
        c = next(color)
    
        lonmin, lonmax, latmin, latmax = domains.domains[k]
    
        shape = geo.gpd_from_domain(lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax)
    
        shape.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color=c) 
    
        ax.text(shape.centroid.x, shape.centroid.y, k, color=c, transform=ccrs.PlateCarree())

    ax.coastlines(resolution='10m') 

    return f

# small utility function to get the geographical domain of 
# an xarray dataset or dataarray 

def get_domain(dset, lon_name='lon', lat_name='lat'): 
    """
    get the lon and lat domain of an xarray dataset or dataarray 

    

    Parameters
    ----------
    dset : xarray dataset or dataarray
        The input dataset or dataarray
    lon_name : str, optional
        The name of the longitude variable, by default 'lon'
    lat_name : str, optional
        The name of the latitude variable, by default 'lat'
    """
    
    lon_min = float(dset[lon_name].data[0])
    lon_max = float(dset[lon_name].data[-1])
    
    lat_min = float(dset[lat_name].data[0])
    lat_max = float(dset[lat_name].data[-1]) 
    
    domain = [lon_min, lon_max, lat_min, lat_max]

    return domain

def extract_domain(dset, domain, lon_name='lon', lat_name='lat'):
    """
    small utility function to extract a domain from a dataset or dataarray

    [extended_summary]

    Parameters
    ----------
    dset : xarray dataset or dataarray
        The input dataset
    domain : list
        The domain [lon_min, lon_max, lat_min, lat_max]
    lon_name : str, optional
        The name of the longitude variable, by default 'lon'
    lat_name : str, optional
        The name of the latitude variable, by default 'lat'
    """
    
    dset = dset.sel({lon_name:slice(*domain[:2]), lat_name:slice(*domain[2:])})
    
    return dset