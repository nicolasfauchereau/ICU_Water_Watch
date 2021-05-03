"""
The `geo` module contains all functions related to reading, manipulating and working with geometries
"""

def make_mask_from_polygon(polygon, lon, lat, wrap_lon=False): 
    """
    make a mask (xarray.DataArray) from a shapely polygon and 
    vectors of longitudes and latitudes

    assumes the shapely geometry is in Lat / Lon (WGS84, EPSG4326)

    Parameters
    ----------
    polygon : shapely Polygon
        The geometry to use to create the mask 
    lon : numpy 1D array
        Vector of longitudes 
    lat : numpy 1D array
        Vector of latitudes
    wrap_lon : bool, optional
        whether or not to wrap the longitudes, by default False

    Returns
    -------
    xarray.DataArray
        xarray.DataArray with the mask (1 inside geometry, np.nan elsewhere)
    """
    import numpy as np 
    import regionmask
    
    poly = regionmask.Regions([polygon])
    mask = poly.mask(lon, lat, wrap_lon=wrap_lon)
    mask = mask.where(np.isnan(mask), 1)
    return mask

def make_mask_from_gpd(dset, gpd_dataframe, lon_name='lon', lat_name='lat', subset=True, insert=True, domain_buffer=0.1, shape_buffer=0.1, mask_name=None): 
    """
    [summary]

    [extended_summary]

    Parameters
    ----------
    dset : [type]
        [description]
    gpd_dataframe : [type]
        [description]
    lon_name : str, optional
        [description], by default 'lon'
    lat_name : str, optional
        [description], by default 'lat'
    subset : bool, optional
        [description], by default True
    insert : bool, optional
        [description], by default True
    domain_buffer : float, optional
        [description], by default 0.1
    shape_buffer : int, optional
        [description], by default 100
    shape_buffer_unit : str, optional
        [description], by default 'km'
    mask_name : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    
    import numpy as np 
    import regionmask
    
    # region from geopandas 
    region = regionmask.from_geopandas(gpd_dataframe)
    
    if subset: # if subset is True, we also subset the input dataset given the region_mask
        
        lon_min = region.bounds_global[0] - domain_buffer
        lon_max = region.bounds_global[2] + domain_buffer
        lat_min = region.bounds_global[1] - domain_buffer
        lat_max = region.bounds_global[3] + domain_buffer
        
        dset = dset.sel({lon_name:slice(lon_min, lon_max), lat_name:slice(lat_min, lat_max)})
    
    # create a mask 
    
    mask = region.mask(dset, lon_name=lon_name, lat_name=lat_name)
    
    # make sure that the values are either np.NaN or 1.
    
    mask = mask.where(np.isnan(mask), 1)
    
    if insert: # if insert is True, we return the original (or subsetted dataset with the mask)
        
        if mask_name is not None:
            
            dset[mask_name] = mask 
        
        else: 
            
            dset['mask'] = mask
            
        return dset 
    
    else: 
        
        return mask 

def read_shapefiles(dpath=None, filename=None, crs=4326, merge=False, buffer=None):
    
    """
    read shapefiles 

    read a shapefile, set the CRS if not set, and optionally merge 

    Returns
    -------
    [type]
        [description]s
    """
    
    
    import pathlib
    import geopandas as gpd 
    
    polygons = gpd.read_file(pathlib.Path(dpath).joinpath(filename))
    
    if polygons.crs is None: 
        if crs is not None: 
            polygons = polygons.set_crs(f'EPSG:{crs}')
        else: 
            raise ValueError(f"The shapefile {filename} in {str(dpath)} has no CRS and not CRS was provided (argument `crs` empty)")
        
    if merge: 
        
        polygons.loc[:,'domain'] = 'domain'
        
        # this is necessary to avoid the regionmask error: "'numbers' must be numeric" 
        # when trying to create a mask from the region ... 
        
        polygons = polygons.dissolve(by='domain', as_index=False).drop('domain', axis=1)
        
        if buffer is not None: 
            
            # first we reproject in UTM 
            
            # buffered = polygons.to_crs('EPSG:3857')
            
            buffered = polygons.buffer(buffer)
            
            # buffered = buffered.to_crs('EPSG:4326')
            
            polygons.loc[:,'geometry'] = buffered
            
    return polygons 

def shift_geom(shift, gdataframe):
    """
    shift the geometries contained in a geodataframe

    used to 'shift' a geopandas.GeoDataFrame so that 
    the longitudes go from 0 to 360 

    Parameters
    ----------
    shift : int
        The number of longitudes to shift (for the above, use -180)
    gdataframe : geopandas.GeoDataFrame
        The input geopandas.GeoDataFrame with the geometries to shift

    Returns
    -------
    geopandas.GeoDataFrame
        The output geopandas.GeoDataFrame
    """

    import geopandas as gpd
    from shapely.geometry import LineString
    from shapely.ops import split
    from shapely.affinity import translate
    
    shift -= 180
    
    border = LineString([(shift,90),(shift,-90)])

    moved_geom = []
    splitted_geom = []
    
    for row in gdataframe["geometry"]:
        splitted_geom.append(split(row, border))

    for element in splitted_geom:
            items = list(element)
            for item in items:
                minx, miny, maxx, maxy = item.bounds
                if minx >= shift:
                    moved_geom.append(translate(item, xoff=-180-shift))
                else:
                    moved_geom.append(translate(item, xoff=180-shift))

    # got `moved_geom` as the moved geometry
    # must specify CRS here, needs to be 4326 (WGS84)
    moved_geom_gdf = gpd.GeoDataFrame({"geometry": moved_geom}, crs="EPSG:4326")

    return moved_geom_gdf


def make_point_buffer_gdf(lon, lat, radius=2000, radius_unit='km'):
    """
    take a point (lon and lat), apply a buffer (defined by kw `radius`)
    and return a geopandas dataframe with the resulting polygon 
    
    to be used e.g. before `utils.make_mask_from_gpd()`

    Parameters
    ----------
    lon : float
        longitude of the point in degrees (WGS84)
    lat : float
        latitude of the point in degrees (WGS84)
    radius : int or float, optional
        The radius for the buffer, by default 2000
    radius_unit : str, optional
        The radius unit, by default 'km', but can also be 'm' (meters)
        or 'degrees'

    Returns
    -------
    [type]
        [description]
    """
    
    from shapely.geometry import Point
    import geopandas as gpd 
    import regionmask
    
    # transform lat and lon into a Point geometry 
    point = Point(lon, lat)
    
    # casts into a GeoDataFrame 
    point_gdf = gpd.GeoDataFrame([], geometry=[point]) 
    
    # set the crs, we assume here that lon and lat are passed into degree, i.e. WGS84, EPSG4326 
    point_gdf = point_gdf.set_crs('EPSG:4326') 
    
    if radius_unit in ['km','m']: 
        point_gdf = point_gdf.to_crs('EPSG:3395') 
        if radius_unit == 'km': 
            # if in km, we multiply by 1000.
            radius *= 1e3 
        # apply the radius 
        area = point_gdf.buffer(radius)
        area = area.to_frame(name='geometry')
        # transform back into EPSG 4326 
        area = area.to_crs('EPSG:4326')
    elif radius_unit == 'degrees': 
        area = point_gdf.buffer(radius)
    
    # now checks the bounds, and make sure it goes from 0 to 360 
    if ((area.bounds.minx < 0).values[0]) or ((area.bounds.maxx < 0).values[0]):
        area = area.to_crs(crs="+proj=longlat +ellps=WGS84 +pm=-180 +datum=WGS84 +no_defs")
        area = shift_geom(-180., area)
        
    # return area, which is a GeoDataFrame, use utils.make_mask_from_gpd (and set buffer to None)
    return area 

def gpd_from_domain(lonmin=None, lonmax=None, latmin=None, latmax=None, crs='4326'): 
    """
    creates a geopandas dataframe with a rectangular domain geometry from 
    min and max longitudes and latitudes

    can be called using gpd_from_domain(*[lonmin, lonmax, latmin, latmax])
    
    can be passed to get_one_GCM() or get_GCMs() as a `mask` keyword argument
 
    Parameters
    ----------
    lonmin : float, optional
        min longitude, by default None
    lonmax : float, optional
        max longitude, by default None
    latmin : float, optional
        min latitude, by default None
    latmax : float, optional
        max latitude, by default None
    crs : str, optional
        The coordinate reference system, by default '4326'

    Returns
    -------
    [type]
        [description]
    """

    
    from shapely.geometry import Polygon
    import geopandas as gpd 
    
    # make the box 
    
    shape = Polygon(((lonmin, latmin), (lonmax, latmin), (lonmax, latmax), (lonmin, latmax), (lonmin, latmin)))  
    
    shape_gpd = gpd.GeoDataFrame([], geometry=[shape])
    
    # set the CRS 
    
    shape_gpd = shape_gpd.set_crs(f'EPSG:{crs}')
    
    return shape_gpd

