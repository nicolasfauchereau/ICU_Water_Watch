import os 
import sys
import matplotlib

import pathlib

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np 
import pandas as pd
import geopandas as gpd
from cartopy import crs as ccrs

from matplotlib import pyplot as plt 
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import palettable

# import local utils and geo module, for interpolation and masking

from . import utils
from . import geo

### some top level definition 

# default domain to set_extent 

domain = [125, 240, -35, 25] 

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


def make_gridlines(ax, lons=None, lats=None, lon_step=5, lat_step=5, left_labels=True, bottom_labels=True): 
    """
    make gridlines (for a geographical plot with cartopy)

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

    if lons is None: 

        lons = np.arange(-180,180+lon_step,lon_step)

    if lats is None: 

        lats = np.arange(-90, 90+lat_step, lat_step)

    gl = ax.gridlines(draw_labels=True, linestyle=':', xlocs=lons, ylocs=lats)
    
    gl.top_labels = False
    
    gl.right_labels = False
    
    if not(left_labels): 
        
        gl.left_labels = False
        
    if not(bottom_labels): 
        
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
    
    title = f"Percentile value for the last {ndays} days"
    
    ax.set_title("") # to get rid of the default title
    
    ax.text(0.99, 0.94, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    
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
        
    # defines the thresholds and labels for the colorbar ticks here
    
    dict_levels = {}
    
    dict_levels[30] = {}
    dict_levels[30]['thresholds'] = [-100, 10, 20, 40, 100, 200, 330, 500]
    dict_levels[30]['cbar_ticklabels'] = ['< 10 mm', '10 ??? 20', '20 ??? 40', '40 ??? 100', '100 ??? 200', '250 ??? 330', '>330 mm']

    dict_levels[60] = {}
    dict_levels[60]['thresholds'] = [-100, 20, 40, 80, 200, 400, 660, 1000]
    dict_levels[60]['cbar_ticklabels'] = ['< 20 mm', '20 ??? 40', '40 ??? 80', '80 ??? 200', '200 ??? 400', '400 ??? 660', '>660 mm'] 
    
    dict_levels[90] = {}
    dict_levels[90]['thresholds'] = [-100, 30, 60, 120, 300, 600, 990, 1500]
    dict_levels[90]['cbar_ticklabels'] = ['< 30 mm', '30 ??? 60', '60 ??? 120', '120 ??? 300', '300 ??? 600', '600 ??? 990', '>990 mm']
    
    dict_levels[180] = {}
    dict_levels[180]['thresholds'] = [-100, 40, 80, 160, 400, 800, 1320, 1700]
    dict_levels[180]['cbar_ticklabels'] = ['< 40 mm', '40 ??? 80', '80 ??? 160', '160 ??? 400', '400 ??? 800', '800 ??? 1320', '>1320 mm'] 
    
    dict_levels[360] = {}
    dict_levels[360]['thresholds'] = [-100, 120, 240, 480, 1200, 2400, 3960, 4500]
    dict_levels[360]['cbar_ticklabels'] = ['< 120 mm', '120 ??? 240', '240 ??? 480', '480 ??? 1200', '1200 ??? 2400', '2400 ??? 3960', '>3960 mm']  
        
    # colors     
    # hexes = ['#8c510a', '#d8b365', '#f6e8c3', '#FFFFFF', '#c7eae5', '#5ab4ac', '#01665e', '#01665e']
    hexes = ['#243494', '#243494', '#225ea8', '#1d91bf', '#42b6c4', '#7fcdbb', '#c7e9b4', '#edf9b1']
    hexes.reverse()

    # get the thresholds and cbar_ticklabels 
    
    thresholds = dict_levels[ndays]['thresholds']
    cbar_ticklabels = dict_levels[ndays]['cbar_ticklabels']

    # ticks locations             
    ticks_marks = np.diff(np.array(thresholds)) / 2.

    ticks = [thresholds[i] + ticks_marks[i] for i in range(len(thresholds) - 1)]

    # arguments for the colorbar 

    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}

    cmap = matplotlib.colors.ListedColormap(hexes, name='accumulations')
    
    # plot starts here ---------------------------------------------------------------------------------
    
    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im = dataarray.plot.contourf(ax=ax, levels=thresholds, transform=ccrs.PlateCarree(), add_colorbar=False, cmap=cmap)

    cbar_ax = ax.axes.inset_axes([0.85, 0.4875, 0.025, 0.42])

    cb = plt.colorbar(im, cax=cbar_ax, **cbar_kwargs)
    
    cb.ax.minorticks_off()
    
    # plots the ticklabels 
    
    cbar_ax.set_yticklabels(cbar_ticklabels)
    
    ax.coastlines(resolution='10m')
    
    if gridlines: 
    
        make_gridlines(ax=ax, lon_step=20, lat_step=10)
    
    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)

    # title 

    ax.set_title('')
    
    # title = f"Last {ndays} days\ncumulative rainfall (mm)"
    title = f"Cumulative rainfall (mm)\n{ndays} days to {last_day:%d %b %Y}"
    
    ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    # copyright notice 
    
    ax.text(0.0065, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    # insert the logo 
    
    # insert_logo(ax=ax)
    
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

    title = f"Last {ndays} days cumulative rainfall anomalies (mm)\nto {last_day:%d %b %Y}" 

    ax.text(0.99, 0.94, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    
    # copyright notice 
    
    ax.text(0.0065, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))
    
    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    # insert the NIWAlogo 
    
    # insert_logo(ax=ax)

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
        
    # defines the levels if the number of days
    
    levels_dict = {}
    levels_dict[30] = np.arange(0, 30 + 5, 5)
    levels_dict[60] = np.arange(0, 60 + 5, 5) 
    levels_dict[90] = np.arange(0, 90 + 10, 10)
    levels_dict[180] = np.arange(0, 180 + 20, 20)
    levels_dict[360] = np.arange(0, 360 + 30, 30)
    
    levels = levels_dict[ndays]
    
    cmap = matplotlib.cm.BrBG_r
    
    cmap = cmap_discretize(cmap, len(np.diff(levels)))
    
    b = [f"< {levels[1]}"]
    a = [f"> {levels[-2]}"]
    m = [f"{levels[i]} - {levels[i + 1]}" for i in range(1, len(levels) - 2)]
    
    cbar_ticklabels = b + m + a
    
    ticks_marks = np.diff(np.array(levels)) / 2.
    
    ticks = [levels[i] + ticks_marks[i] for i in range(len(levels) - 1)]
    
    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}
    
    # plot starts here --------------------------------------------------------------------------------- 
    
    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im = dataarray.plot.contourf(ax=ax, levels=levels, transform=ccrs.PlateCarree(), add_colorbar=False, cmap=cmap)
    
    # cbar_ax = ax.axes.inset_axes([0.88, 0.35, 0.025, 0.55])
    
    cbar_ax = ax.axes.inset_axes([0.85, 0.4875, 0.025, 0.42])
    
    cb = plt.colorbar(im, cax=cbar_ax, **cbar_kwargs)
    
    cb.ax.minorticks_off()
    
    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)
    
    cbar_ax.set_yticklabels(cbar_ticklabels)
    
    ax.coastlines(resolution='50m')
    
    if gridlines: 
    
        make_gridlines(ax=ax, lon_step=20, lat_step=10)

    ax.set_title('', fontsize=13, color='k') # get rid of default title

    title = f"Number of dry days over the last {ndays} days\nto {last_day:%d %b %Y}"
    
    ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    # copyright notice 
    
    ax.text(0.0065, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    # insert the NIWA logo 
    
    # insert_logo(ax=ax)
    
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
        
    elif ndays in [90, 180]: 
        
        levels = np.arange(0, ndays + 10, 10)
        
    else: 
        
        levels = np.arange(0, ndays + 30, 30)
    
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
    
    title = f"Days since last rain, {ndays} days period\nto {last_day:%d %b %Y}"
    
    ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    # copyright notice 
    
    ax.text(0.0065, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    # insert the NIWA logo 
    
    # insert_logo(ax=ax)
    
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
    
    levels_dict = {}
    levels_dict[30] = np.arange(0, 30 + 5, 5)
    levels_dict[60] = np.arange(0, 60 + 5, 5) 
    levels_dict[90] = np.arange(0, 90 + 10, 10)
    levels_dict[180] = np.arange(0, 180 + 20, 20)
    levels_dict[360] = np.arange(0, 360 + 30, 30)
    
    levels = levels_dict[ndays]
    
    cmap = matplotlib.cm.BrBG_r
    
    cmap = cmap_discretize(cmap, len(np.diff(levels)))
    
    b = [f"< {levels[1]}"]
    a = [f"> {levels[-2]}"]
    m = [f"{levels[i]} - {levels[i + 1]}" for i in range(1, len(levels) - 2)]
    
    cbar_ticklabels = b + m + a
    
    ticks_marks = np.diff(np.array(levels)) / 2.
    
    ticks = [levels[i] + ticks_marks[i] for i in range(len(levels) - 1)]
    
    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}
    
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

    ax.set_title("")
    
    if varname == 'dry_days': 
        title = f"{country_name}: Number of dry days, {ndays} days period\nto {last_day:%d %b %Y}"
        ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    elif varname == 'days_since_rain': 
        title = f"{country_name}: Days since last rain, {ndays} days period\nto {last_day:%d %b %Y}"
        ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    elif varname == 'wet_days': 
        title = title = f"{country_name}: Number of wet days, {ndays} days period\nto {last_day:%d %b %Y}"
        ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        
    # copyright notice 
    
    ax.text(0.01, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])    

    # insert the NIWA logo 
    
    # insert_logo(ax=ax) 
                
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
        
        filename = f"{utils.sanitize_name(country_name)}_{mask_type}_{varname}_{ndays}nbdays_to_{last_day:%Y-%m-%d}.png"
        
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

    # rgbs = ['#8a0606', '#fc0b03','#fcf003','#ffffff', '#0335ff']
    rgbs = ['#F04E37', '#F99D1C','#FFDE40','#FFFFFF', '#33BBED']
    
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

    title = f"Water Stress ({ndays} days to {last_day:%d %b %Y})\n(aligned to \"EAR\" alert levels)"
    
    ax.set_title("") # to get rid of the default title
    
    ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    
    # copyright notice 
    
    ax.text(0.0065, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    # insert the logo 
    
    # insert_logo(ax=ax)
    
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

    title = f"US Drought Monitor\n{ndays} days to {last_day:%d %b %Y}"

    # ax.set_title(title, fontsize=13, color='k')
    
    ax.set_title("") # to get rid of the default title
    
    ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    # copyright notice 
    
    ax.text(0.0065, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    # insert the NIWAlogo 
    
    # insert_logo(ax=ax)
    
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

    # colors for each level 

    rgbs = ['#F04E37', '#F99D1C','#FFDE40','#FFFFFF', '#33BBED']
    
    # EAR Watch ticklabels, hard coded 
    
    cbar_ticklabels = ["Severely dry (< 5%)",'Seriously dry (< 10%)', 'Warning (< 25%)', 'Near or Wetter', 'Seriously wet (> 90%)']
    
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

    EEZ.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='0.4', linewidth=0.8)

    gl = ax.gridlines(draw_labels=True, linestyle=':', xlocs=xlocs, ylocs=ylocs, crs=ccrs.PlateCarree())

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    f.patch.set_facecolor('white')
        
    title = f"{country_name}: \"EAR\" Watch alert levels\n{ndays} days to {last_day:%d %b %Y}"
    
    ax.set_title(title, fontsize=13, color='k')

    f.patch.set_facecolor('white')

    # copyright notice 
    
    ax.text(0.02, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])    

    # insert the logo 
    
    # insert_logo(ax=ax)

    country_name_file = utils.sanitize_name(country_name)
        
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

    EEZ.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='0.4', linewidth=0.8)

    gl = ax.gridlines(draw_labels=True, linestyle=':', xlocs=xlocs, ylocs=ylocs, crs=ccrs.PlateCarree())

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    f.patch.set_facecolor('white')
        
    title = f"{country_name}: US Drought Monitor\n{ndays} days to {last_day:%d %b %Y}"
    
    ax.set_title(title, fontsize=13, color='k')

    f.patch.set_facecolor('white')

    # copyright notice 
    
    ax.text(0.02, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])    
            
    if fpath is None: # if no path, then the path is the current directory 
        
        fpath = pathlib.Path.cwd()
        
    else: 
            
        fpath = pathlib.Path(fpath)
            
        if not fpath.exists():
            
            fpath.mkdir(parents=True)
    
    # build the file name 
    
    filename = f"{utils.sanitize_name(country_name)}_{mask_name}_USDM_{ndays}nbdays_to_{last_day:%Y-%m-%d}.png"
        
    # save the figure 
    
    f.savefig(fpath.joinpath(filename), dpi=200, bbox_inches='tight')
        
    if close: 
        
        plt.close(f)

def map_SPI_Pacific(dset, varname='SPI', mask=None, geoms=None, domain=[125, 240, -35, 25], fpath=None, close=True, gridlines=False): 

    # get the last date and the number of days from the dataset 
    
    last_day, ndays = get_attrs(dset) 

    dataarray = dset[varname].squeeze()
    
    # if mask_EEZs is True, we multiply the percentage of score by the EEZs mask 

    if (mask is not None) and (mask in dset.data_vars): 
        
        dataarray = dataarray * dset[mask]

    # here hard-coded thresholds, colors and labels 

    thresholds = [-2.5, -2, -1.5, -1, 1, 1.5, 2, 2.5]

    rgbs = ['#F04E37', '#F99D1C', '#FFDE40', '#ffffff', '#96ceff', '#4553bf', '#09146b']

    ticks_marks = np.diff(np.array(thresholds)) / 2.

    ticks = [thresholds[i] + ticks_marks[i] for i in range(len(thresholds) - 1)]

    cbar_ticklabels = ['extremely dry','severely dry','moderately dry',' ', 'moderately wet','severely wet','extremely wet']

    cmap = matplotlib.colors.ListedColormap(rgbs, name='SPI')

    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}

    # plot starts here 
    
    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im = dataarray.plot.contourf(ax=ax, levels=thresholds, cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False)

    # adds the colorbar axes as insets 

    cbar_ax = ax.axes.inset_axes([0.80, 0.5225, 0.025, 0.38])
    
    # plots the colorbar in these axes 
    
    cb = plt.colorbar(im, cax=cbar_ax, **cbar_kwargs)

    # removes the minor ticks 

    cb.ax.minorticks_off() 

    # plots the ticklabels 
    
    cbar_ax.set_yticklabels(cbar_ticklabels)
    
    # removes the ticks 
    
    cb.ax.tick_params(size=0)

    ax.coastlines(resolution='10m')

    if geoms is not None: 
        
        add_geom(ax=ax, geoms=geoms)

    if gridlines: 
        
        make_gridlines(ax=ax, lon_step=20, lat_step=10)

    title = f"Standardized Precipitation Index (SPI)\n{ndays} days to {last_day:%d %b %Y}"
    
    ax.set_title("") # to get rid of the default title
    
    ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
    
    # copyright notice 
    
    ax.text(0.0065, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))
    
    ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    # insert the logo 
    
    # insert_logo(ax=ax)
    
    f.patch.set_facecolor('white')
    
    if fpath is not None: 
        
        if type(fpath) != pathlib.PosixPath: 
            
            fpath = pathlib.Path(fpath)

        f.savefig(fpath.joinpath(f"GPM_IMERG_SPI_Pacific_{ndays}days_to_{last_day:%Y-%m-%d}.png"), dpi=200, bbox_inches='tight')
    
    if close: 
        
        plt.close(f)

def map_SPI(dset_sub_country, world_coastlines, country_coastline, EEZ, varname='SPI', mask_name='mask_EEZ', country_name='Fiji', fpath=None, close=True): 

    thresholds = [-2.5, -2, -1.5, -1, 1, 1.5, 2, 2.5]

    rgbs = ['#F04E37', '#F99D1C', '#FFDE40', '#ffffff', '#96ceff', '#4553bf', '#09146b']

    ticks_marks = np.diff(np.array(thresholds)) / 2.

    ticks = [thresholds[i] + ticks_marks[i] for i in range(len(thresholds) - 1)]

    cbar_ticklabels = ['extremely dry','severely dry','moderately dry',' ', 'moderately wet','severely wet','extremely wet']

    cmap = matplotlib.colors.ListedColormap(rgbs, name='SPI')

    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}

    # extracts the number of days and last day 

    last_day, ndays = get_attrs(dset_sub_country)

    # determine boundaries 

    lon_min = np.floor(dset_sub_country.lon.data.min())
    lon_max = np.ceil(dset_sub_country.lon.data.max())
    lat_min = np.floor(dset_sub_country.lat.data.min())
    lat_max = np.ceil(dset_sub_country.lat.data.max())

    # make xlocs and ylocs 
    
    xlocs = np.linspace(lon_min, lon_max + 1, 5, endpoint=True)
    ylocs = np.linspace(lat_min, lat_max + 1, 5, endpoint=True)
    
    xlocs[xlocs > 180] -= 360 # fix the longitudes to go from -180 to 180 (for plotting)

    # extract the dataarray and apply the mask 
    if mask_name is not None: 

        dataarray = (dset_sub_country[varname] * dset_sub_country[mask_name]).squeeze()

    else: 

        dataarray = dset_sub_country[varname].squeeze()


    f, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})

    im_arr  = dataarray.plot.contourf(ax=ax, levels=thresholds, cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False, extend='both')

    # cbar_ax = ax.axes.inset_axes([1.05, 0.225, 0.035, 0.6])

    # plots the colorbar in these axes 

    cb = plt.colorbar(im_arr, **cbar_kwargs)

    cb.ax.tick_params(size=0)

    # plots the ticklabels 

    cb.ax.set_yticklabels(cbar_ticklabels)

    world_coastlines.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), lw=1, color='0.8')

    country_coastline.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), lw=1, color='k')

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    add_geom(ax, geoms=EEZ)

    gl = ax.gridlines(draw_labels=True, linestyle=':', xlocs=xlocs, ylocs=ylocs, crs=ccrs.PlateCarree())

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}

    title = f"{country_name}: Standardized Precipitation Index (SPI)\n{ndays} days to {last_day:%d %b %Y}"

    ax.set_title(title, fontsize=13, color='k')

    ax.text(0.02, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    
    if fpath is None: # if no path, then the path is the current directory 
        
        fpath = pathlib.Path.cwd()
        
    else: 
            
        fpath = pathlib.Path(fpath)
            
        if not fpath.exists():
            
            fpath.mkdir(parents=True)
    
    # build the file name 
    
    filename = f"{utils.sanitize_name(country_name)}_{mask_name}_SPI_{ndays}nbdays_to_{last_day:%Y-%m-%d}.png"
        
    # save the figure 
    
    f.savefig(fpath.joinpath(filename), dpi=200, bbox_inches='tight', facecolor='w')
        
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

def plot_heatmap(mat, year = 2021, start_month=3, period='monthly', n_steps=5, cumsum=False, title=None, cmap='Oranges'): 
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
    
    # munging on month_abbr to account for period straddling 2 calendar years 
    
    month_abbr = list(month_abbr)
    
    month_abbr = month_abbr + month_abbr[1:]
    
    if period == 'monthly':
    
        xtick_labels = month_abbr[start_month: start_month + n_steps]
    
    elif period == 'seasonal': 
        
        xtick_labels = [f"{month_abbr[start_month + i]} - {month_abbr[start_month + i + 2]}" for i in range(n_steps)]
    
    if cumsum: 
        
        mat = mat.cumsum(axis=0)
    
        ytick_labels = [f'< {x}' for x in range(10, 100 + 10, 10)] 
        
    else: 
        
        ytick_labels = [f'{x} to {x+10}' for x in range(0, 100, 10)]
        
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

def map_MME_forecast(probs_mean, \
                          varname='precip',
                          step=1,
                          pct_dim='percentile', \
                          pct=None, \
                          comp='below', \
                          interp=True, \
                          mask=None, \
                          geoms=None, \
                          domain=None, \
                          fpath=None, \
                          fname=None, \
                          close=True, \
                          gridlines=False): 



    # munging on the month abbreviations to account for periods straddling 2 years 
    
    from calendar import month_name
    
    month_name = list(month_name)
    
    month_name = month_name + month_name[1:]
    
    # period taken from the dataset  

    period = probs_mean.attrs['period']

    # some checks 
    
    if (not(pct_dim in ['tercile','quartile', 'decile','percentile'])) or (not(pct_dim in probs_mean.dims)): 
        
        raise ValueError(f"{pct_dim} not valid, should be in ['tercile','decile','percentile'] and be a dimension in probs_mean") 
        
    # get month and year of the forecast (initial month)
    
    # valid time
        
    month = probs_mean.time.to_index().month[0]
    
    year = probs_mean.time.to_index().year[0]
    
    # take the step 
    
    probs_mean = probs_mean.sel(step=step).squeeze()

    # period label 
    
    if period == 'monthly': 

        period_label = month_name[month + step]

    elif period == 'seasonal': 
        
        period_label = month_name[month + step - 2] + " - " + month_name[month + step]
        
    # interpolation 
    
    if interp: 
    
        probs_mean = utils.interp(probs_mean)
    
    # get the percentile bins  
    
    percentile_bins = probs_mean.attrs['pct_values']
    
    # digitize, so we can get the corresponding category along the tercile, decile or percentile dimension 
    
    max_cat = np.digitize(pct / 100, percentile_bins)
    
    # calculate the probability of being either below or above the given percentile 
    
    if comp == 'above': 
        
        ptot = probs_mean.sel({pct_dim:slice(max_cat+1, None)}).sum(pct_dim)
        
    elif comp == 'below':  
        
        ptot = probs_mean.sel({pct_dim:slice(None, max_cat)}).sum(pct_dim)
    
    dops = {}
    dops['below'] = '<'
    dops['above'] = '>'
        
    if (mask is not None) and (type(mask) == gpd.geodataframe.GeoDataFrame): 
        
        ptot = geo.make_mask_from_gpd(ptot, mask, subset=False, insert=True, mask_name='EEZ')
        
        ptot = ptot[varname] * ptot['EEZ']

    # set the parameters for plotting 

    thresholds = [0, 50, 60, 70, 80, 90, 100]
    
    # hexes = ['#a6dba0', '#d9f0d3', '#f7f7f7', '#e7d4e8', '#c2a5cf', '#9970ab', '#762a83']
    # hexes = ['#ffffff', '#f6e8c3', '#dfc27d', '#bf812d','#ecb256', '#c18b36', '#543005']

    ticks_marks = np.diff(np.array(thresholds)) / 2.

    ticks = [thresholds[i] + ticks_marks[i] for i in range(len(thresholds) - 1)]

    # cbar_ticklabels = ["< 25%", "25-50%", "50-60%", "60-70%", "70-80%", "80-90%", "> 90%"]
    
    cbar_ticklabels = ["< 50%", "50-60%", "60-70%", "70-80%", "80-90%", "> 90%"]
    
    # cmap = matplotlib.colors.ListedColormap(hexes, name='probabilities')
    
    cmap = cmap_discretize(palettable.scientific.sequential.Bilbao_6.mpl_colormap, 6)
    
    # starts the plot
    
    f, ax = plt.subplots(figsize=(14,8), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)})

    # uncomment this if needs to add contours 

    # for i, ct in enumerate(contours): 

    #     fc = ptot.plot.contour(ax=ax, x='lon',y='lat', levels=[ct], colors=contours_colors[i], linewidths=0.7, transform=ccrs.PlateCarree())

    ff = ptot.plot.contourf(ax=ax, x='lon',y='lat', levels=thresholds, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)

    cbar_kwargs={'shrink':0.5, 'pad':0.01, 'extend':'neither', 'drawedges':True, 'ticks':ticks, 'aspect':15}

    cbar_ax = ax.axes.inset_axes([0.9, 0.525, 0.025, 0.38])

    cb = plt.colorbar(ff, cax=cbar_ax, **cbar_kwargs)

    cb.ax.minorticks_off()

    cbar_ax.set_yticklabels(cbar_ticklabels)

    ax.coastlines(resolution='10m')

    if geoms is not None:
        
        add_geom(ax=ax, geoms=geoms)
    
    if gridlines: 
        
        make_gridlines(ax=ax, lon_step=20, lat_step=10)

    # set the title
    
    ax.set_title("")
    
    title = f"Probability of rainfall {dops[comp]} {pct}th percentile\n{period_label}"

    ax.text(0.99, 0.95, title, fontsize=13, fontdict={'color':'k'}, bbox=dict(facecolor='w', edgecolor='w'), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    # adds the copyright sentence 

    ax.text(0.0065, 0.02, u'\u00A9'+" NIWA", transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='w'), fontdict=dict(color='0.4'))

    # add the NIWAlogo 
    
    # insert_logo(ax=ax)

    # set the bottom line, which indicates the valid time for the forecast (be it monthly or seasonal)

    # if period is not None: 
        
    #     ax.text(0.99, 0.02, f"{period} forecast, valid to end {valid_time:%Y-%m}", fontsize=10, fontdict={'color':'0.4'}, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    # else: 

    #     ax.text(0.99, 0.02, f"valid {valid_time:%Y-%m}", fontsize=10, fontdict={'color':'0.4'}, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    if domain is not None: 
    
        ax.set_extent(domain, crs = ccrs.PlateCarree())
    
    f.patch.set_facecolor('white')
    
    if fname is None: 
        
        if period is not None: 
        
            fname = f"C3S_{period}_MME_{comp}_{pct}_{year}_{month}_init_{period_label.replace(' - ','_')}.png"
        
        else: 
             
            fname = f"C3S_MME_{comp}_{pct}_{year}_{month}_init_{period_label.replace(' - ','_')}.png"
            
    if fpath is not None: 
        
        if type(fpath) != pathlib.PosixPath: 
            
            fpath = pathlib.Path(fpath)

        if period is not None: 

            f.savefig(fpath.joinpath(fname), dpi=200, bbox_inches='tight')
            
        else: 
            
            f.savefig(fpath.joinpath(fname), dpi=200, bbox_inches='tight')
    
    if close: 
        
        plt.close(f)

def map_MME_probabilities(probs_mean, \
                          nsteps=5, \
                          pct_dim='percentile', \
                          pct=None, \
                          comp='below', \
                          cmap=None, 
                          varname='precip', \
                          contours=[50, 70], \
                          contours_colors=['r','#a3231a'], \
                          shape=None, \
                          domain=None, 
                          mask=None): 

    from calendar import month_abbr

    # munging on the month abbreviations to account for periods straddling 2 years 
    
    month_abbr = list(month_abbr)
    
    month_abbr = month_abbr + month_abbr[1:]

    # some checks 
    
    if (not(pct_dim in ['tercile','decile','percentile'])) or (not(pct_dim in probs_mean.dims)): 
        
        raise ValueError(f"{pct_dim} not valid, should be in ['tercile','decile','percentile'] and be a dimension in probs_mean") 
        
    # get month and year of the forecast (initial month)
    
    month = probs_mean.time.to_index().month[0]
    year = probs_mean.time.to_index().year[0]
    
    # get the percentile bins  
    
    percentile_bins = probs_mean.attrs['pct_values']
    
    # digitize, so we can get the corresponding category along the tercile, decile or percentile dimension 
    
    max_cat = np.digitize(pct / 100, percentile_bins)
    
    # calculate the probability of being either below or above the given percentile 
    
    if comp == 'above': 
        
        ptot = probs_mean.sel({pct_dim:slice(max_cat+1, None)}).sum(pct_dim)
        
    elif comp == 'below':  
        
        ptot = probs_mean.sel({pct_dim:slice(None, max_cat)}).sum(pct_dim)
    
    # if not colormap was passed, we get sensible defaults depending on whether we 
    # want probabilities of being below a percentile values (warm colors) or above (cool colors)
    # i.e. we assume the variable to plot is precip 
    
    dops = {}
    dops['below'] = '<'
    dops['above'] = '>'
     
    if cmap is None: 
        
        if comp == 'below': 
            
            if palettable:
            
                cmap = palettable.scientific.sequential.Bilbao_20.mpl_colormap
                
            else: 
                
                cmap = plt.cm.Oranges
    
        elif comp == 'above': 
            
            if palettable: 
            
                cmap = palettable.scientific.sequential.Davos_20_r.mpl_colormap
            
            else: 
                
                cmap = plt.cm.Blues

    # we get the maximum probability from ptot 
    
    max_prob = float(ptot.max(['step','lat','lon']).squeeze()[varname].data)
        
    # if the contours that are passed are over the maximum probabilities, we define some sensible defaults
    
    if max(contours) > max_prob: 
        
        contours_l = [(int(max_prob * 0.80) // 10 * 10) -10,  int(max_prob * 0.80) // 10 * 10]
                
        print(f"\nWARNING: some of the passed contours [{','.join(list(map(str, contours)))}] are over the maximum probability ({max_prob:4.2f}), using [{','.join(list(map(str, contours_l)))}] instead\n")
        
        contours = contours_l 
    
    steps = probs_mean['step'].data
    
    # plot 
    
    f, axes = plt.subplots(nrows=nsteps, figsize=(4,14), subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)})
    
    for i, step in enumerate(steps[:nsteps]): 
        
        ax = axes[i]
        
        ax.set_extent(domain, crs=ccrs.PlateCarree())
        
        p = ptot.sel({'step':step})[varname].squeeze()
        
        if (mask is not None) and (mask in probs_mean.data_vars):
            
            p = p * probs_mean[mask] 
            
        ff = p.plot.contourf(ax=ax, x='lon',y='lat', levels=np.arange(10,100,5), add_colorbar=True, transform=ccrs.PlateCarree(), \
                             cbar_kwargs={'shrink':0.9, 'label':'%', 'aspect':10, 'pad':0.01, 'extend':'neither'}, cmap=cmap)
        
        for i, ct in enumerate(contours): 
            
            fc = p.plot.contour(ax=ax, x='lon',y='lat',levels=[ct], colors=contours_colors[i], linewidths=0.7, transform=ccrs.PlateCarree())
                
        if step != nsteps: ax.set_xlabel('')
            
        if step != nsteps: ax.set_xticks([])

        ax.coastlines(resolution='10m',lw=0.5)

        if shape is not None: 
        
            shape.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='0.4',lw=0.5)

        if domain is not None: 
        
            ax.set_extent(domain, crs=ccrs.PlateCarree())
            
        ax.set_title(f'prob. {varname} {dops[comp]} {pct}th perc., {month_abbr[month + step]} {year}\n [ECMWF, Meteo-France, UKMO, DWD, CMCC, NCEP, JMA, ECCC]', fontsize=11)        
    
    f.set_figwidth(8)
    
    f.set_figheight(8 * (nsteps / 2.3))
    
    return f

def plot_virtual_station(df, station_name=None, lon=None, lat=None): 
    
    # some parameters we'll use later 
    
    ndays = len(df) 
    last_day = f"{df.index[-1]:%d %B %Y}"
    sums = df.sum()
    
    # initialise the figure 
    
    f = plt.figure(figsize=(12,6))

    # first axes: time-series of past N days daily rainfall, and climatology
    
    ax1 = f.add_axes([0.1,0.25,0.7,0.65])

    ax1.plot(df['climatology'].index, df['climatology'], color='g', label="")
    ax1.plot(df['observed'].index, df['observed'], color='b', label="")

    ax1.fill_between(df['climatology'].index, 0, df['climatology'], color='g', alpha=0.6, label='climatology (GPM/IMERG)')
    ax1.fill_between(df['observed'].index, 0, df['observed'], color='b', alpha=0.6, label='estimated (GPM/IMERG)')

    [l.set_rotation(90) for l in ax1.xaxis.get_ticklabels()]
    [l.set_fontsize(12) for l in ax1.xaxis.get_ticklabels()]
    [l.set_fontsize(12) for l in ax1.yaxis.get_ticklabels()]

    ax1.legend(fontsize=12, loc=2)

    ax1.set_xlim(df.index[0], df.index[-1])

    ax1.set_ylim(0, None)

    ax1.set_ylabel("mm", fontsize=12)
    
    ax1.grid(ls=':')

    if station_name is not None: 
        
        ax1.set_title(f"Last {ndays} days to {last_day}, GPM-IMERG virtual station for {station_name} [{lon:5.3f}E, {lat:5.3f}S]")
        
    else: 
        
        ax1.set_title(f"Last {ndays} days to {last_day}, GPM-IMERG virtual station for coordinates [{lon:5.3f}E, {lat:5.3f}S]")
    
    
    # second axes: cumulative rainfall as barplots, with percentage of normal over the past N days

    ax2 = f.add_axes([0.8,0.25,0.14,0.65])
    ax2.bar(np.arange(0.5,2.5), sums.values, width=0.7, color=['b','g'], alpha=0.6, align='center')
    ax2.set_xticks([0.5, 1.5])
    ax2.set_xticklabels(['obs.', 'clim.'], fontsize=12)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("mm", fontsize=12)
    ax2.yaxis.set_label_position("right")

    ax2.set_title(f"{np.divide(*sums.values) * 100:4.1f} % of normal", fontdict={'weight':'bold'})

    [l.set_fontsize(12) for l in ax2.yaxis.get_ticklabels()]

    return f

def insert_logo(logo_path='./logo', x=-0.06, y=0.88, width=0.25, ax=None): 
    """
    insert a logo into axes 

    insert the NIWAlogo into the given axes 

    Parameters
    ----------
    logo_path : str, optional
        The path to the logo image (png), by default './logo'
    x : float, optional
        x coordinates (in axes units), by default -0.06
    y : float, optional
        y coordinates (in axes units), by default 0.88
    width : float, optional
        The width (in axes units), by default 0.25
    ax : [type], optional
        The axes where to insert, by default None
    """
    
    
    impath = os.path.dirname(sys.modules['ICU_Water_Watch'].__file__)
    
    impath = pathlib.Path(impath).joinpath('logo')
            
    logo = plt.imread(impath.joinpath("NIWA_CMYK_Hor.png"), format='png')
    
    aspect = logo.shape[0] / logo.shape[1]
    
    axin = ax.inset_axes([x,y, width, width * aspect],transform=ax.transAxes)    # create new inset axes in data coordinates
    
    axin.imshow(logo)
    
    axin.axis('off')
    
def get_hexes(cmap): 
    """
    get a list of hex values from a matplotlib colormap 

    Parameters
    ----------
    cmap : a matplotlib colormap 
        The input matplotlib colormap 

    Returns
    -------
    list
        A list of hex values as strings (e.g. '#ffffff' for white)
    """
    l = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
        l.append(matplotlib.colors.rgb2hex(rgb))
    l = np.array(l)

    hexes = (np.unique(l)[::-1]).tolist()

    return hexes 

def hex_to_rgb(value):
    """
    converts a hex value to RGB 

    e.g. hex_to_rgb('#ffffff') -> (255, 255, 255)

    Parameters
    ----------
    value : string
        The hex value as string

    Returns
    -------
    tuple
        tuple with (R, G, B) values (0 to 255)
    """
    
    
    value = value.lstrip('#')
    
    lv = len(value)
    
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    """
    (R,G,B) to hex value

    convert a tuple of RGB values to hex code
    
    example: 
    
    >> rgb_to_hex((255,255,255))
    >> '#ffffff'

    Parameters
    ----------
    rgb : tuple
        tuple with (R, G, B) values (0 to 255)

    Returns
    -------
    str
        hex value as string
    """
    return '#%02x%02x%02x' % rgb
