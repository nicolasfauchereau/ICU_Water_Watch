import matplotlib

import pathlib

from datetime import datetime, timedelta

import numpy as np 
import pandas as pd
import geopandas as gpd
from cartopy import crs as ccrs

from matplotlib import pyplot as plt 
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import palettable

### some top level definition 

# domain to set_extent 

domain = [125, 240, -35, 25] # domain to set_extent 

# dictionnary for the precipitation accumulation plots 

dict_max = {}

dict_max[1] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[2] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[3] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[4] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[5] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[6] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[7] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[8] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[9] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[10] = {30:{'vmax':1000, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[11] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

dict_max[12] = {30:{'vmax':1600, 'step':100}, \
                60:{'vmax':1600, 'step':100}, \
                90:{'vmax':1600, 'step':100}, \
                180:{'vmax':2400, 'step':200}, \
                360:{'vmax':4600, 'step':400}}

# dictionnary for the anomalies 

dict_anoms_max = {}
dict_anoms_max[30] = {'vmax':300, 'step':50}
dict_anoms_max[60] = {'vmax':400, 'step':50}
dict_anoms_max[90] = {'vmax':600, 'step':100}
dict_anoms_max[180] = {'vmax':1000, 'step':200}
dict_anoms_max[360] = {'vmax':1400, 'step':200}

### some common utility functions 

def get_attrs(dset): 
    """
    return (in order) the last day and the number of days
    in a dataset
    
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

def add_geom(ax=None, geoms=None):
    """
    adds geometries to a geoAxes 
    obtained e.g. with `plt.subplots(subplots_kws={'projection':ccrs.PlateCarree(central_longitude=180.)}))`

    Parameters
    ----------
    ax : geoAxes, optional
        the axes in which to plot the geometries, by default None
    geoms : geopandas GeoDataFrame or list of GeoDataFrame
        geometries to plot, by default None
    """
    
    # single geodataframe
    
    if type(geoms is gpd.geodataframe.GeoDataFrame): 
        
        geoms.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='0.4', linewidth=0.8)
    
    # list of geodataframe
     
    elif (type(geoms) is list) and (np.alltrue([type(geom) is gpd.geodataframe.GeoDataFrame for geom in geoms])): 
        
        for geom in geoms:
            
            geom.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='0.4', linewidth=0.8)
            
    # if not, there's a problem 
    else:
        
        raise ValueError(f"the geometry passed is not a geopandas GeoDataFrames or a LIST of GeoDataFrames")


def make_gridlines(ax, lon_step=5, lat_step=5, left_labels=True, bottom_labels=True): 
    """
    make gridlines (for a geographical plot with cartopy)

    [extended_summary]

    Parameters
    ----------
    ax : GeoAxesSubplot
        The specific axes in which to draw the gridlines
    lonstep : int, optional
        The step in longitude, by default 5
    latstep : int, optional
        The step in latitude, by default 5
    
    Usage
    ----- 
    
    >> fg = dset['varname'].plot.pcolormesh(x='lon',y='lat',col='step')
    >> [make_gridlines(x, lon_step=20, lat_step=10) for x in fg.axes[0]]
    
    """
    
    gl = ax.gridlines(draw_labels=True, linestyle=':', \
        xlocs=np.arange(-180, 180 + lon_step, lon_step), \
            ylocs=np.arange(-90, 90 + lat_step, lat_step), \
                crs=ccrs.PlateCarree())
    
    gl.top_labels = False
    gl.right_labels = False
    
    if not left_labels: 
        gl.left_labels = False
        
    if not bottom_labels: 
        gl.bottom_labels = False
    

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    
    colors_rgba = cmap(colors_i)
    
    indices = np.linspace(0, 1., N+1)
    
    cdict = {}
    
    for ki,key in enumerate(('red','green','blue')):
        
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def colorbar_index(ncolors, cmap, ticklabels=None, fontsize=15, color='gray', label='Percentile', cbar_kwargs=None):
    
    cmap = cmap_discretize(cmap, ncolors)
    
    mappable = cm.ScalarMappable(cmap=cmap)
    
    mappable.set_array([])
    
    mappable.set_clim(-0.5, ncolors+0.5)
    
    if cbar_kwargs is not None: 
        colorbar = plt.colorbar(mappable, **cbar_kwargs)
    else: 
        colorbar = plt.colorbar(mappable)
    
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    
    colorbar.set_label(label, fontsize=fontsize, color=color)
    
    if ticklabels is not None: 
        
        colorbar.set_ticklabels(ticklabels)
    
    else: 
        
        colorbar.set_ticklabels(np.arange(1, ncolors + 1), fontsize=fontsize, color=color)

    colorbar.ax.tick_params(labelsize=fontsize, labelcolor=color)

def map_decile(dset, varname='pctscore', mask=None, cmap=None, geoms=None, ticklabels=None, fpath=None, close=True, gridlines=False): 
    
    last_day, ndays = get_attrs(dset)
    
    dataarray = dset[varname].squeeze()
    
    if (mask is not None) and (mask in dset.data_vars): 
        
        dataarray = dataarray * dset[mask]
     
    if ticklabels is None: 
        
        ticklabels=["< 10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "> 90"]
    
    cbar_kwargs={'shrink':0.7, 'pad':0.025 , 'label':'percentile'}
    
    if cmap is None and palettable: 
        
        hex_colors = palettable.scientific.diverging.Roma_10.hex_colors
        hex_colors[4] = hex_colors[5] = '#ffffff'
        cmap = matplotlib.colors.ListedColormap(hex_colors, name='roma')
    
    else: 
        
        cmap = plt.cm.RdBu_r
    
    # plot starts here ---------------------------------------------------------------------------------    
        
    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.Mercator(central_longitude=180)})
    
    ax.contourf(dataarray['lon'], dataarray['lat'], dataarray, transform=ccrs.PlateCarree(), cmap=cmap)
    
    # dataarray.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap)
    
    colorbar_index(ncolors=10, cmap=cmap, ticklabels=ticklabels, cbar_kwargs=cbar_kwargs, color='k', fontsize=13)    

    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)

    ax.coastlines(resolution='10m', color='k', lw=1)

    # overlay grid lines 
    
    if gridlines: 

        make_gridlines(ax=ax, lon_step=20, lat_step=10)
    
    # ax.set_title(title, fontsize=13, color='k')
    
    title = f"Percentile value for the last {ndays} days\n to {last_day:%d %B %Y} [UTC]"
    
    ax.set_title("") # to get rid of the default title
    
    ax.text(0.99, 0.94, title, fontsize=13, fontdict={'color':'green'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    
    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    f.patch.set_facecolor('white')

    if fpath is not None: 
        
        if type(fpath) != pathlib.PosixPath: 
            
            fpath = pathlib.Path(fpath)

        f.savefig(fpath.joinpath(f"pctscore_{ndays}days_to_{last_day:%Y-%m-%d}.png"), dpi=200, bbox_inches='tight') 

    if close: 

        plt.close(f)

def map_precip_accum(dset, varname='precipitationCal', mask=None, geoms=None, cmap=None, fpath=None, close=True, gridlines=False): 

    last_day, ndays = get_attrs(dset) 
    
    dataarray = dset[varname].squeeze()

    if (mask is not None) and (mask in dset.data_vars): 
        
        dataarray = dataarray * dset[mask] 
        
    # min, max and steps are defined in a dictionnary, with keys being the number of days
    
    vmax = dict_max[last_day.month][ndays]['vmax']
    step = dict_max[last_day.month][ndays]['step']
    
    levels = np.arange(0,vmax + step,step)
        
    if cmap is None and palettable: 
        
        cmap = palettable.scientific.sequential.Davos_20_r.mpl_colormap
    
    else: 
        
        cmap = plt.cm.viridis
        
    cbar_kwargs={'shrink':0.7, 'pad':0.01, 'label':'mm', 'ticks':levels}
    
    # plot starts here ---------------------------------------------------------------------------------
    
    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im = dataarray.plot.contourf(ax=ax, levels=levels, transform=ccrs.PlateCarree(), add_colorbar=False, cmap=cmap)

    cbar_ax = ax.axes.inset_axes([0.91, 0.275, 0.025, 0.6])

    cb = plt.colorbar(im, cax=cbar_ax, **cbar_kwargs)
    
    cb.ax.minorticks_off()
    
    ax.coastlines(resolution='10m')
    
    if gridlines: 
    
        make_gridlines(ax=ax, lon_step=20, lat_step=10)
    
    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)

    ax.set_title('')
    
    title = f"Last {ndays} days cumulative rainfall (mm)\n(to {last_day:%d %B %Y} [UTC])"
    
    ax.text(0.99, 0.94, title, fontsize=13, fontdict={'color':'green'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    f.patch.set_facecolor('white')

    if fpath is not None: 
        
        if type(fpath) != pathlib.PosixPath: 
            
            fpath = pathlib.Path(fpath)

        f.savefig(fpath.joinpath(f"precip_accumulation_{ndays}days_to_{last_day:%Y-%m-%d}.png"), dpi=200, bbox_inches='tight')
    
    if close: 
        
        plt.close(f)


def map_precip_anoms(dset, varname='anoms', mask=None, cmap=None, geoms=None, fpath=None, close=True, gridlines=False): 
    """
    """

    last_day, ndays = get_attrs(dset) 
    
    dataarray = dset[varname].squeeze()

    if (mask is not None) and (mask in dset.data_vars): 
        
        dataarray = dataarray * dset[mask] 
            
    vmax = dict_anoms_max[ndays]['vmax']
    
    step = dict_anoms_max[ndays]['step']
    
    levels = np.arange(-vmax,vmax + step,step)
    
    cbar_kwargs={'shrink':0.7, 'pad':0.01, 'extend':'both', 'ticks':np.arange(-vmax, vmax+step, step), 'drawedges':True, 'label':'mm'}

    if cmap is None and palettable: 
        cmap = palettable.colorbrewer.diverging.BrBG_11.mpl_colormap
    else: 
        cmap = plt.cm.RdBu
    
    # plot starts here ---------------------------------------------------------------------------------
    
    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im = dataarray.plot.contourf(ax=ax, levels=levels, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    
    cbar_ax = ax.axes.inset_axes([0.9, 0.275, 0.025, 0.6])

    cb = plt.colorbar(im, cax=cbar_ax, **cbar_kwargs)
    
    cb.ax.minorticks_off()

    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)

    ax.coastlines(resolution='10m')
    
    if gridlines: 
    
        make_gridlines(ax=ax, lon_step=20, lat_step=10)
        
    ax.set_title("") # to get rid of default title

    title = f"Last {ndays} days cumulative rainfall anomalies (mm)\nto {last_day:%d %B %Y} [UTC]" 

    ax.text(0.99, 0.94, title, fontsize=13, fontdict={'color':'green'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    
    ax.set_extent(domain, crs = ccrs.PlateCarree())

    f.patch.set_facecolor('white')
    
    if fpath is not None: 
        
        if type(fpath) != pathlib.PosixPath: 
            
            fpath = pathlib.Path(fpath)

        f.savefig(fpath.joinpath(f"precip_anomalies_{ndays}days_to_{last_day:%Y-%m-%d}.png"), dpi=200, bbox_inches='tight')
    
    if close: 
        
        plt.close(f)
        
def map_dry_days_Pacific(dset, varname='dry_days', mask=None, cmap=None, geoms=None, fpath=None, close=True, gridlines=False): 
    
    last_day, ndays = get_attrs(dset)

    dataarray = dset[varname].squeeze()
    
    if (mask is not None) and (mask in dset.data_vars):
        
        dataarray = dataarray * dset[mask]
        
    # defines the levels 
    
    if ndays in [30, 60]: 
        
        levels = np.arange(0, ndays + 5, 5)
        
    else: 
        
        levels = levels = np.arange(0, ndays + 10, 10)
            
    # colorbar keyword arguments 
    
    cbar_kwargs = {'shrink':0.7, 'pad':0.01, 'extend':'max', 'drawedges':True, 'label':'days', 'ticks':levels}
    
    # colormap 

    if cmap is None and palettable: 
        
        cmap = palettable.colorbrewer.sequential.Oranges_9.mpl_colormap
    
    else:
         
        cmap = cm.Oranges
    
    # plot starts here --------------------------------------------------------------------------------- 
    
    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im = dataarray.plot.contourf(ax=ax, levels=levels, transform=ccrs.PlateCarree(), add_colorbar=False, cmap=cmap)
    
    cbar_ax = ax.axes.inset_axes([0.9, 0.3, 0.025, 0.55])
    
    cb = plt.colorbar(im, cax=cbar_ax, **cbar_kwargs)
    
    cb.ax.minorticks_off()
    
    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)
    
    ax.coastlines(resolution='50m')
    
    if gridlines: 
    
        make_gridlines(ax=ax, lon_step=20, lat_step=10)

    ax.set_title('', fontsize=13, color='k') # get rid of default title

    title = f"Number of dry days over the last {ndays} days\n(to {last_day:%d %B %Y} [UTC])"
    
    ax.text(0.99, 0.94, title, fontsize=13, fontdict={'color':'green'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    f.patch.set_facecolor('white')
    
    if fpath is not None: 
        
        if type(fpath) != pathlib.PosixPath: 
            
            fpath = pathlib.Path(fpath)

        f.savefig(fpath.joinpath(f"nb_dry_days_{ndays}days_to_{last_day:%Y-%m-%d}.png"), dpi=200, bbox_inches='tight')
    
    if close: 
        
        plt.close(f)

def map_days_since_rain_Pacific(dset, varname='days_since_rain', mask=None, cmap=None, geoms=None, fpath=None, close=True, gridlines=False): 
    
    last_day, ndays = get_attrs(dset)
    
    dataarray = dset[varname].squeeze()
    
    if (mask is not None) and (mask in dset.data_vars): 
        
        dataarray = dataarray * dset[mask]
        
    # defines the levels 
    
    if ndays in [30, 60]: 
        
        levels = np.arange(0, ndays + 5, 5)
        
    else: 
        
        levels = levels = np.arange(0, ndays + 10, 10)
    
    # colorbar keyword arguments 
    
    cbar_kwargs = {'shrink':0.7, 'pad':0.01, 'extend':'max', 'drawedges':True, 'label':'days', 'ticks':levels}
    
    # colormap 

    if cmap is None and palettable: 
        
        cmap = palettable.colorbrewer.sequential.Oranges_9.mpl_colormap
    
    else:
         
        cmap = cm.Oranges
        
    # plot starts here ---------------------------------------------------------------------------------
    
    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im = dataarray.plot.contourf(ax=ax, levels=levels, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)

    cbar_ax = ax.axes.inset_axes([0.9, 0.3, 0.025, 0.55])
    
    cb = plt.colorbar(im, cax=cbar_ax, **cbar_kwargs)
    
    cb.ax.minorticks_off()
   
    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)
    
    ax.coastlines(resolution='10m')
    
    if gridlines:
    
        make_gridlines(ax=ax, lon_step=20, lat_step=10)

    ax.set_title("") # get rid of default title

    title = f"Days since last rain (>= 1 mm/day): \n{ndays} days period to {last_day:%d %B %Y} [UTC]"
    
    ax.text(0.99, 0.94, title, fontsize=13, fontdict={'color':'green'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    f.patch.set_facecolor('white')
    
    if fpath is not None: 
        
        if type(fpath) != pathlib.PosixPath: 
            
            fpath = pathlib.Path(fpath)

        f.savefig(fpath.joinpath(f"days_since_rain_{ndays}days_to_{last_day:%Y-%m-%d}.png"), dpi=200, bbox_inches='tight')
    
    if close: 
        
        plt.close(f)

        
def map_dry_days(dset, world_coastlines, country_coastline, EEZ, varname='dry_days', \
    mask_type='mask_EEZ', country_name='Fiji', cbar_label='number of days', cbar_kwargs=None, \
        figsize=None, cmap=None, path=None, filename=None, close=True, lonlat=None): 
    
    # get the last date and the number of days from the dataset 
    
    last_day, ndays = get_attrs(dset)
    
    levels = np.arange(0, ndays + (ndays // 10), (ndays // 10))
    
    # determine boundaries 
    
    lon_min = np.floor(dset.lon.data.min())
    lon_max = np.ceil(dset.lon.data.max())
    lat_min = np.floor(dset.lat.data.min())
    lat_max = np.ceil(dset.lat.data.max())
    
    # make xlocs and ylocs 
    
    xlocs = np.linspace(lon_min, lon_max + 1, 5, endpoint=True)
    ylocs = np.linspace(lat_min, lat_max + 1, 5, endpoint=True)
    
    xlocs[xlocs > 180] -= 360 # fix the longitudes to go from -180 to 180 (for plotting)
    
    if figsize is None: 
        
        figsize = (13, 8)
    
    if cmap is None: 
        
        cmap = palettable.colorbrewer.sequential.Oranges_9.mpl_colormap
    
    if cbar_kwargs is not None: 
         
        cbar_kwargs['label'] = cbar_label
        
    else: 
            
        cbar_kwargs = {'shrink':0.7, 'pad':0.01, 'extend':'max', 'drawedges':True, 'label':cbar_label}

    # plot ---------------------------------------------------------------------------------------------------------------

    f, ax = plt.subplots(figsize=figsize, subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
        
    (dset[varname] * dset[mask_type]).plot.contourf(ax=ax, levels=levels, \
                                                                  cmap=cmap, \
                                                                  transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs)


    # cbar_ax = f.axes[-1]

    world_coastlines.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='0.8', lw=0.8)

    country_coastline.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k', lw=0.8)

    EEZ.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='steelblue', lw=1)
    
    if lonlat is not None: 
    
        ax.plot(lonlat[0],lonlat[1], marker='*', color='k', markersize=15,
                alpha=0.7, transform=ccrs.PlateCarree())
        ax.plot(lonlat[0],lonlat[1], marker='*', color='blue', markersize=10,
                transform=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linestyle=':', xlocs=xlocs, ylocs=ylocs, crs=ccrs.PlateCarree())

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    f.patch.set_facecolor('white')
    
    if varname == 'dry_days': 
        title = f"{country_name}: GPM-IMERG, number of dry days (< 1 mm/day)\nover the past {ndays} days to {last_day:%d %B %Y} [UTC]"
        ax.set_title(title, fontsize=13, color='k')
    elif varname == 'days_since_rain': 
        title = f"{country_name}: GPM-IMERG, days since last rain (>= 1 mm/day)\nover the past {ndays} days to {last_day:%d %B %Y} [UTC]"
        ax.set_title(title, fontsize=13, color='k')   
    elif varname == 'wet_days': 
        title = f"{country_name}: GPM-IMERG, number of wet days (>= 1 mm/day)\nover the past {ndays} days to {last_day:%d %B %Y} [UTC]"
        ax.set_title(title, fontsize=13, color='k')
        
    print(title)

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])    

    special_chars = [' / ', '/',' ','%',':', '&']

    for special_char in special_chars: 
        
        country_name_file = country_name.replace(special_char,'_')
        
    if path is None: # if no path, then the path is the current directory 
        
        path = pathlib.Path.cwd()
        
    else: 
        
        if type(path) is not pathlib.PosixPath: 
            
            path = pathlib.Path(path)
            
        if not path.exists(): 
            
            path.mkdir(parents=True)
        
    if filename is not None: 
        
        f.savefig(path.joinpath(filename), dpi=200, bbox_inches='tight')
    
    else: 
        
        filename = f"{country_name_file}_{mask_type}_{varname}_{ndays}nbdays_to_{last_day:%Y-%m-%d}.png"
        
        f.savefig(path.joinpath(filename), dpi=200, bbox_inches='tight')
        
    if close: 
        
        plt.close(f)
        
def map_EAR_Watch_Pacific(dset, varname='pctscore', mask=None, geoms=None, fpath=None, close=True, gridlines=False): 

    # get the last date and the number of days from the dataset 
    
    last_day, ndays = get_attrs(dset) 

    dataarray = dset[varname].squeeze()
    
    # if mask_EEZs is True, we multiply the percentage of score by the EEZs mask 

    if (mask is not None) and (mask in dset.data_vars): 
        
        dataarray = dataarray * dset[mask]

    # here hard-coded thresholds, colors and labels 

    thresholds = [0, 5, 10, 25, 90.01, 100]

    rgbs = ['#8a0606', '#fc0b03','#fcf003','#ffffff', '#0335ff']
    
    ticks_marks = np.diff(np.array(thresholds)) / 2.
    
    ticks = [thresholds[i] + ticks_marks[i] for i in range(len(thresholds) - 1)]

    cbar_ticklabels = ["Severely dry (< 5%)",'Seriously dry (< 10%)', 'Warning (< 25%)', 'Near or Wetter', 'Seriously wet (> 90%)']

    # arguments for the colorbar 
    
    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}
    
    cmap = matplotlib.colors.ListedColormap(rgbs, name='EAR Watch')
    
    # plot starts here 
    
    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im = dataarray.plot.contourf(ax=ax, levels=thresholds, cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False)

    # adds the colorbar axes as insets 

    cbar_ax = ax.axes.inset_axes([0.80, 0.525, 0.025, 0.38])
    
    # plots the colorbar in these axes 
    
    cb = plt.colorbar(im, cax=cbar_ax, **cbar_kwargs)

    # removes the minor ticks 

    cb.ax.minorticks_off() 

    # plots the ticklabels 
    
    cbar_ax.set_yticklabels(cbar_ticklabels)

    ax.coastlines(resolution='10m')

    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)

    if gridlines: 
        
        make_gridlines(ax=ax, lon_step=20, lat_step=10)

    title = f"GPM-IMERG, \"EAR\" Watch alert levels\n{ndays} days to {last_day:%d %B %Y} [UTC]"

    # ax.set_title(title, fontsize=13, color='k')
    
    ax.set_title("") # to get rid of the default title
    
    ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'green'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    f.patch.set_facecolor('white')
    
    if fpath is not None: 
        
        if type(fpath) != pathlib.PosixPath: 
            
            fpath = pathlib.Path(fpath)

        f.savefig(fpath.joinpath(f"EAR_Watch_Pacific_{ndays}days_to_{last_day:%Y-%m-%d}.png"), dpi=200, bbox_inches='tight')
    
    if close: 
        
        plt.close(f)


def map_USDM_Pacific(dset, mask=None, geoms=None, fpath=None, close=True, gridlines=False): 

    # here hard-coded thresholds, colors and labels

    thresholds = [0, 2, 5, 10, 20, 30, 100]
    
    rgbs = ['#8a0606', '#fc0b03','#fc9003','#ffd08a','#ffeb0f','#ffffff']
    
    ticks_marks = np.diff(np.array(thresholds)) / 2.
    
    ticks = [thresholds[i] + ticks_marks[i] for i in range(len(thresholds) - 1)]
    
    cmap = matplotlib.colors.ListedColormap(rgbs, name='USDM')
    
    cbar_ticklabels = ['D4 (Exceptional Drought)', 'D3 (Extreme Drought)', 'D2 (Severe Drought)', 'D1 (Moderate Drought)', 'D0 (Abnormally Dry)', 'None']
    
    # arguments for the colorbar

    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}

   # get the last date and the number of days from the dataset 
    
    last_day, ndays = get_attrs(dset) 

    # if mask_EEZs is True, we multiply the percentage of score by the EEZs mask 

    dataarray = dset['pctscore'].squeeze() 

    if (mask is not None) and (mask in dset.data_vars): 
        
        dataarray = dataarray * dset[mask]

    # plot starts here

    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im  = dataarray.plot.contourf(ax=ax, levels=thresholds, cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False)

    cbar_ax = ax.axes.inset_axes([0.78, 0.53, 0.025, 0.38])
    
    cb = plt.colorbar(im, cax=cbar_ax, **cbar_kwargs)
    
    cb.ax.minorticks_off()

    # cbar_ax.set_yticks(ticks)
    
    cbar_ax.set_yticklabels(cbar_ticklabels)

    ax.coastlines(resolution='10m')
    
    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)
        
    if gridlines: 
        
        make_gridlines(ax=ax, lon_step=20, lat_step=10)

    title = f"\"US Drought Monitor\" (see https://droughtmonitor.unl.edu/)\n{ndays} days to {last_day:%d %B %Y} [UTC]"

    # ax.set_title(title, fontsize=13, color='k')
    
    ax.set_title("") # to get rid of the default title
    
    ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'green'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    f.patch.set_facecolor('white')
    
    if fpath is not None: 
        
        if type(fpath) != pathlib.PosixPath: 
            
            fpath = pathlib.Path(fpath)

        f.savefig(fpath.joinpath(f"USDM_Pacific_{ndays}days_to_{last_day:%Y-%m-%d}.png"), dpi=200, bbox_inches='tight')
    
    if close: 
        
        plt.close(f)

def map_EAR_Watch(dset, world_coastlines, country_coastline, EEZ, varname='pctscore', mask_name='mask_EEZ', country_name='Fiji', fpath=None, close=True): 
    
    # get the last date and the number of days from the dataset 
        
    last_day, ndays = get_attrs(dset)
    
    # make sure dset is squeezed now 
    
    dset = dset.squeeze()
    
    # determine boundaries 
    
    lon_min = np.floor(dset.lon.data.min())
    lon_max = np.ceil(dset.lon.data.max())
    lat_min = np.floor(dset.lat.data.min())
    lat_max = np.ceil(dset.lat.data.max())
    
    # make xlocs and ylocs 
    
    xlocs = np.linspace(lon_min, lon_max + 1, 5, endpoint=True)
    ylocs = np.linspace(lat_min, lat_max + 1, 5, endpoint=True)
    
    xlocs[xlocs > 180] -= 360 # fix the longitudes to go from -180 to 180 (for plotting)
    
    # EAR Watch thresholds, hard coded 
    
    thresholds = [0, 5, 10, 25, 90.01, 100]
    
    # EAR Watch ticklabels, hard coded 
    
    cbar_ticklabels = ["Severely dry (< 5%)",'Seriously dry (< 10%)', 'Warning (< 25%)', 'Near or Wetter', 'Seriously wet (> 90%)']
    
    # colors for each level 
    
    rgbs = ['#8a0606', '#fc0b03','#fcf003','#ffffff', '#0335ff']
    
    # automatically defines tickmarks 
    
    ticks_marks = np.diff(np.array(thresholds)) / 2.
    
    ticks = [thresholds[i] + ticks_marks[i] for i in range(len(thresholds) - 1)]
    
    # colormap from RGB list 
    
    cmap = matplotlib.colors.ListedColormap(rgbs, name='EAR Watch')

    # keywords arguments for the colorbar
    
    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}

    # plot starts here

    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    (dset[varname] * dset[mask_name]).plot.contourf(ax=ax, levels=thresholds, cmap=cmap, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs)

    cbar_ax = f.axes[-1]

    cbar_ax.set_yticklabels(cbar_ticklabels)

    world_coastlines.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k', lw=0.8)

    country_coastline.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k', lw=0.8)

    EEZ.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='steelblue', lw=1)

    gl = ax.gridlines(draw_labels=True, linestyle=':', xlocs=xlocs, ylocs=ylocs, crs=ccrs.PlateCarree())

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    f.patch.set_facecolor('white')
    
    title = f"{country_name}: GPM-IMERG, \"EAR\" watch advisory levels\n{ndays} days to {last_day:%d %B %Y} [UTC]"
    
    ax.set_title(title, fontsize=13, color='k')

    f.patch.set_facecolor('white')

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])    

    special_chars = [' / ', '/',' ','%',':', ' & ']

    for special_char in special_chars: 
        
        country_name_file = country_name.replace(special_char,'_')
        
    if fpath is None: # if no path, then the path is the current directory 
        
        fpath = pathlib.Path.cwd()
        
    else: 
        
        fpath = pathlib.Path(fpath)
            
        if not fpath.exists(): 
            
            fpath.mkdir(parents=True)
    
    # build the file name 
    
    filename = f"{country_name_file}_{mask_name}_EAR_Watch_{ndays}nbdays_to_{last_day:%Y-%m-%d}.png"
        
    # save the figure 
    
    f.savefig(fpath.joinpath(filename), dpi=200, bbox_inches='tight')
        
    if close: 
        
        plt.close(f)

def map_USDM(dset, world_coastlines, country_coastline, EEZ, varname='pctscore', mask_name='mask_EEZ', country_name='Fiji', fpath=None, close=True): 
   
    # get the last date and the number of days from the dataset 
        
    last_day, ndays = get_attrs(dset)
    
    # make sure dset is squeezed now 
    
    dset = dset.squeeze()    
    
    # determine boundaries 
    
    lon_min = np.floor(dset.lon.data.min())
    lon_max = np.ceil(dset.lon.data.max())
    lat_min = np.floor(dset.lat.data.min())
    lat_max = np.ceil(dset.lat.data.max())
    
    # make xlocs and ylocs 
    
    xlocs = np.linspace(lon_min, lon_max + 1, 5, endpoint=True)
    ylocs = np.linspace(lat_min, lat_max + 1, 5, endpoint=True)
    
    xlocs[xlocs > 180] -= 360 # fix the longitudes to go from -180 to 180 (for plotting)
    
    # EAR Watch thresholds, hard coded 
    
    thresholds = [0, 2, 5, 10, 20, 30, 100]
    
    # EAR Watch ticklabels, hard coded 
    
    cbar_ticklabels = [f"D{i}" for i in range(len(thresholds)-2)]
    
    cbar_ticklabels.reverse() 
    
    cbar_ticklabels = cbar_ticklabels + ['None']
        
    # colors for each level 
    
    rgbs = ['#8a0606', '#fc0b03','#fc9003','#ffd08a','#ffeb0f','#ffffff']
    
    # automatically defines tickmarks 
    
    ticks_marks = np.diff(np.array(thresholds)) / 2.
    
    ticks = [thresholds[i] + ticks_marks[i] for i in range(len(thresholds) - 1)]
    
    # colormap from RGB list 
    
    cmap = matplotlib.colors.ListedColormap(rgbs, name='USDM')

    # keywords arguments for the colorbar
    
    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}

    # plots starts here

    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    (dset[varname] * dset[mask_name]).plot.contourf(ax=ax, levels=thresholds, \
                                                                      cmap=cmap, \
                                                                      transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs)

    cbar_ax = f.axes[-1]

    cbar_ax.set_yticklabels(cbar_ticklabels)

    world_coastlines.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k', lw=0.8)

    country_coastline.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k', lw=0.8)

    EEZ.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='steelblue', lw=1)

    gl = ax.gridlines(draw_labels=True, linestyle=':', xlocs=xlocs, ylocs=ylocs, crs=ccrs.PlateCarree())

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    f.patch.set_facecolor('white')
    
    title = f"{country_name}: GPM-IMERG, \"USDM\" levels\n{ndays} days to {last_day:%d %B %Y} [UTC]"
    
    ax.set_title(title, fontsize=13, color='k')

    f.patch.set_facecolor('white')

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])    

    special_chars = [' / ', '/',' ','%',':', ' & ']

    for special_char in special_chars: 
        
        country_name_file = country_name.replace(special_char,'_')
        
    if fpath is None: # if no path, then the path is the current directory 
        
        fpath = pathlib.Path.cwd()
        
    else: 
            
        fpath = pathlib.Path(fpath)
            
        if not fpath.exists():
            
            fpath.mkdir(parents=True)
    
    # build the file name 
    
    filename = f"{country_name_file}_{mask_name}_USDM_{ndays}nbdays_to_{last_day:%Y-%m-%d}.png"
        
    # save the figure 
    
    f.savefig(fpath.joinpath(filename), dpi=200, bbox_inches='tight')
        
    if close: 
        
        plt.close(f)

        
def plot_eofs(eofs, verif_dset='CMAP'): 
    """
    plots the EOFS held in a xarray.Dataarray (as from appliying eofs from A.J. Dawson      )

    Parameters
    ----------
    eofs : xarray.Dataarray
        The xarray.Dataarray containing the EOFS, see for example: 
        validation/make_obs_PCA.ipynb 
    verif_dset : string 
        The name of the verification dataset ('MSWEP' or 'CMAP' currently)

    Returns
    -------
    matplotlib.Figure instance
        Return the figure
    """
    
    n_eofs = len(eofs.mode)
    
    fg = eofs[f'EOFs_{verif_dset}'].plot.contourf(x='lon',y='lat',col='mode', levels=np.arange(-1, 1 + 0.1, 0.1), \
                         subplot_kws={'projection':ccrs.PlateCarree(central_longitude=180)},\
                          transform=ccrs.PlateCarree(),\
                          cmap=plt.cm.RdBu_r,\
                         cbar_kwargs={'shrink':0.5, 'orientation':'horizontal', 'pad':0.1, 'aspect':50, 'drawedges':True})
    
    [fg.axes[0][i].set_title(f'{verif_dset}, EOF# {i + 1}') for i in range(n_eofs)]; 

    for i, ax in enumerate(fg.axes[0]): 

        gl = ax.gridlines(draw_labels=True, linestyle=':', \
            xlocs=np.arange(-180, 180 + 40, 40), \
                ylocs=np.arange(-90, 90 + 20, 20), \
                    crs=ccrs.PlateCarree())
        gl.top_labels = False
        if i != n_eofs - 1:
            gl.right_labels = False
        if i != 0: 
            gl.left_labels = False
        if i == n_eofs - 1: 
            gl.right_labels = True

    fg.map(lambda: plt.gca().coastlines(resolution='10m', lw=0.5))
    
    fg.cbar.set_label('R')
    
    return fg.fig

def map_quantile_categories(dset, varname='precip', steps=[1,2,3], n_quantiles=3, cbar_ticklabels=None, cmap='viridis', cbar_label=None, domain=None):
    
    cbar_kwargs = {}
    cbar_kwargs['ticks'] = np.arange(n_quantiles) + 0.5
    cbar_kwargs['shrink'] = 0.7
    cbar_kwargs['orientation'] = 'horizontal'
    cbar_kwargs['aspect'] = 50
    
    if cbar_ticklabels is None: 
        
        cbar_ticklabels = np.arange(n_quantiles) + 1
    
    if type(steps) == list or type(steps) == np.ndarray: 
        n_steps = len(steps)
    else: 
        n_steps = 1

    if domain is not None: 
        dset = dset.sel(lon=slice(*domain[0:2]), lat=slice(*domain[2:]))

    if n_steps > 1:
        
        fg = dset[varname].sel(step=steps).squeeze().plot.contourf(x='lon',y='lat', col='step',\
                                    vmin =0, \
                                    vmax = n_quantiles,
                                    levels = n_quantiles + 1,  
                                    cbar_kwargs=cbar_kwargs, \
                                    cmap=cmap, \
                                    subplot_kws={"projection": ccrs.PlateCarree(central_longitude=180)}, \
                                    transform=ccrs.PlateCarree())
        
        fg.map(lambda: plt.gca().coastlines(resolution='10m', lw=0.5))
                    
        for i in range(n_steps): 
            
            if i == 0: 
                
                make_gridlines(fg.axes[0][i], lon_step=40, lat_step=20)
            
            else:
                
                make_gridlines(fg.axes[0][i], lon_step=40, lat_step=20, left_labels=False)     
            
    fg.cbar.ax.set_xticklabels(cbar_ticklabels)
    
    if cbar_label is not None: 
        
        fg.cbar.set_label(cbar_label)
        
        # if cbar_kwargs['orientation'] == 'horizontal': 
        #     fg.cbar.ax.set_xticklabels(cbar_ticklabels)
        # elif cbar_kwargs['orientation'] == 'vertical':
        #     fg.cbar.ax.set_yticklabels(cbar_ticklabels)
        
    return fg.fig, fg.axes

def plot_heatmap(mat, year = 2021, start_month=3, n_steps=5, cumsum=False, title=None, cmap='Oranges'): 
    """
    Plot a heatmap of decile probabilities or cumulative decile probabilities

    [extended_summary]

    Parameters
    ----------
    mat : the dataframe with x-axis = step, and y-axis = decile category
        [description]
    year : int, optional
        The year of the forecast, by default 2021
    start_month : int, optional
        The start month of the forecast (NOT the initial time), by default 3
    n_steps : int, optional
        The number of lead times in months, by default 5
    cumsum : bool, optional
        Whether or not to calculate the cumulative probabilities, by default False
    title : str, optional
        The title for the Plot, by default None
    cmap : str, optional
        The colormap, by default 'Oranges_r'

    Returns
    -------
    Matplotlib Figure and Axes instances
        So call >> f, ax = plot_heatmap(...)
    """
    
    import seaborn as sns
    from calendar import month_abbr
    
    xtick_labels = month_abbr[start_month: start_month + n_steps]
    
    if cumsum: 
        
        mat = mat.cumsum(axis=0)
    
        ytick_labels = [f'< {x}%' for x in range(10, 100 + 10, 10)] 
        
    else: 
        
        ytick_labels = [f'{x} to {x+10}%' for x in range(0, 100, 10)]
        
        cmap = cmap.replace('_r','')
    
    f, ax = plt.subplots(figsize=(6,5))
    
    sns.heatmap(mat/100, ax=ax, cmap=cmap, annot=True, fmt="2.0%", cbar=False)
    
    if title is not None: 
        ax.set_title(title)
    
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels, rotation=0)
    ax.set_xlabel(str(year))
    ax.set_ylabel("<--- wetter                drier --->", loc='center')

    ax.axhline(4, color='r')
    ax.axhline(6, color='steelblue')
    
    return f, ax 