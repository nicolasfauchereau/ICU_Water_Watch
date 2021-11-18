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