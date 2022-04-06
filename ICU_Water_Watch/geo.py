"""
The `geo` module contains all functions related to reading, manipulating and working with geometries
"""

from . import utils

def dissolve(*geoms, buffer=0.05):
    
    """
    to use instead of calling the geopandas method `dissolve` which fails on 
    the EEZs shapefile with some weird error, usage
    
    >> EEZs = gpd.read_file('.....shapefiles/EEZs/ICU_geometries0_360_EEZ.shp') 
    >> geoms = EEZs.geometry.values
    >> EEZs_merged = dissolve(*geoms)
    >> EEZs_merged_gdf = gpd.GeoDataFrame({'geometry':[EEZs_merged]}, crs="EPSG:4326")
    >> EEZs_merged_gdf.index = ['EEZ'] 
    >> EEZs_merged_gdf.to_file('.....shapefiles/ICU_geometries0_360_EEZ_merged.shp')
    """
    
    from shapely.ops import cascaded_union
    
    return cascaded_union([
        geom.buffer(buffer) if geom.is_valid else geom.buffer(buffer) for geom in geoms
    ])

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

def make_mask_from_gpd(dset, gpd_dataframe, lon_name='lon', lat_name='lat', subset=True, insert=True, domain_buffer=0.1, mask_name=None): 
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
        
    if merge: # TODO: replace with call to custom function `dissolve`
        
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
    
    import pathlib
    
    if dpath_shapes is None: 
        
        dpath_shapes = pathlib.Path.cwd().parents[2].joinpath('shapefiles')
        
    else: 
        
        if type(dpath_shapes) != pathlib.PosixPath: 
            
            dpath_shapes = pathlib.Path(dpath_shapes)

    if not dpath_shapes.exists(): 
        
        raise ValueError(f"{str(dpath_shapes)} does not exist")
    
    EEZs = read_shapefiles(dpath_shapes.joinpath('EEZs'), filename='ICU_geometries0_360_EEZ.shp')
    
    merged_EEZs = read_shapefiles(dpath_shapes.joinpath('EEZs'), filename='ICU_geometries0_360_EEZ_merged.shp')
    
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
    
    import pathlib

    if dpath_shapes is None: 
        
        dpath_shapes = pathlib.Path.cwd().parents[2].joinpath('shapefiles')
        
    else: 
        
        if type(dpath_shapes) != pathlib.PosixPath: 
            
            dpath_shapes = pathlib.Path(dpath_shapes)

    if not dpath_shapes.exists(): 
        
        raise ValueError(f"{str(dpath_shapes)} does not exist")
    
    coastlines = read_shapefiles(dpath_shapes.joinpath('Coastlines'), filename='ICU_geometries0_360_coastlines.shp')
        
    return coastlines

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
    
    # transform lat and lon into a Point geometry 
    point = Point(lon, lat)
    
    # casts into a GeoDataFrame 
    point_gdf = gpd.GeoDataFrame([], geometry=[point]) 
    
    # set the crs, we assume here that lon and lat are passed into degree, i.e. WGS84, EPSG4326 
    point_gdf = point_gdf.set_crs('EPSG:4326') 
    
    if radius_unit in ['km','m']: 
        point_gdf = point_gdf.to_crs('EPSG:3857') 
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
    
    can be passed e.g. to get_one_GCM() or get_GCMs() as a `mask` keyword argument
 
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

def filter_by_area(shape, min_area = 1000):
    """
    filter a geopandas dataframe (assumed to be in lat / lon) containing
    a MultiPolygon geometry by retaining only Polygons with area >= min_area

    Parameters
    ----------
    shape : geopandas.GeoDataFrame
        The geopandas dataframe
    min_area : float, optional
        The minimum area in square kilometers, by default 1000

    Returns
    -------
    geopandas.GeoDataFrame
        The filtered geopandas dataframe
    """
    
    import geopandas as gpd
    from shapely.geometry.multipolygon import MultiPolygon
    
    # project to pseudo mercator to be able to get area in square km
    
    shape_m = shape.to_crs('EPSG:3857')
    
    Polygons = list(shape_m.geometry.values[0])
    
    Filtered_Polygons = [x for x in Polygons if (x.area / 10**6) >= min_area]
    
    Filtered_Polygons = MultiPolygon(Filtered_Polygons)
    
    new_shape = gpd.geodataframe.GeoDataFrame({'geometry':[Filtered_Polygons]}, crs="EPSG:3857")
    
    new_shape = new_shape.to_crs(crs="+proj=longlat +ellps=WGS84 +pm=-180 +datum=WGS84 +no_defs")
    
    new_shape = shift_geom(-180., new_shape)
    
    new_shape = gpd.geodataframe.GeoDataFrame({'geometry':[MultiPolygon(list(new_shape.geometry.values))]}, crs="EPSG:4326")
    
    return new_shape

def mask_dataset(dset, shape, varname='precip', lat_name='lat', lon_name='lon', domain_buffer=1, coastline_buffer=15, interp_factor=5): 
    
    import numpy as np 
    import regionmask 
    from dask.diagnostics import ProgressBar
    
    # get the bounds (lat and lon) from the shape (Polygon or Multipolygon)
    
    bounds = shape.bounds.values.flatten()
    
    # apply domain_buffer (in degrees)

    domain = [bounds[0] - domain_buffer, bounds[2] + domain_buffer, bounds[1] - domain_buffer, bounds[3] + domain_buffer]

    # extract a rectangular domain from the dataset
    
    dset = dset.sel(lat=slice(*domain[2:]), lon=slice(*domain[:2]))
        
    # apply coastline buffer (expressed in km)

    if coastline_buffer is not None: 
        
        shape_buffer = shape.to_crs('EPSG:3857')

        shape_buffer = shape_buffer.buffer(coastline_buffer*1e3)

        shape_buffer = shape_buffer.to_crs('EPSG:4326')

        shape_buffer = shape_buffer.to_frame(name='geometry')

        shape_buffer = shape_buffer.to_crs(crs="+proj=longlat +ellps=WGS84 +pm=-180 +datum=WGS84 +no_defs")

        # make sure we shift the geometries
        
        shape_buffer = shift_geom(-180., shape_buffer)
    
    else: 

        shape_buffer = shape

    # interpolate the dataset 
    
    dset = utils.interp(dset, interp_factor=interp_factor)
    
    # now get the interpolated lats and lons 
    
    lat, lon = dset[lat_name].data, dset[lon_name].data
    
    # create the mask 
    
    mask = regionmask.mask_geopandas(shape_buffer, lon, lat)
    
    mask = mask.where(np.isnan(mask), 1)
    
    # insert the mask in the interpolated dataset 
    
    dset['mask'] = mask
    
    # count the number of valid grid cells and add that to the mask attributes
    
    dset['mask'].attrs['cells'] = int(dset['mask'].stack(z=('lat','lon')).sum('z'))
    
    # apply the mask 
    
    dset[varname] = dset[varname] * dset['mask']
    
    # compute 
    
    with ProgressBar(): 
        
        dset = dset.compute()
    
    # return the masked dataset 
    
    return dset, domain

def get_shape_bounds(shape):
    
    """
    get the domain (lonmin, lonmax, latmin, latmax)
    from the bounds of a shape (geopandas dataframe)
    containing a column named "geometry"

    Returns
    -------
    list
        list with [lonmin, lonmax, latmin, latmax]
    """
    
    domain = [float(shape.geometry.bounds.minx), float(shape.geometry.bounds.maxx), float(shape.geometry.bounds.miny), float(shape.geometry.bounds.maxy)]
    
    return domain 