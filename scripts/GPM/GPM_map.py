#!/home/nicolasf/mambaforge/envs/ICU/bin/python
# coding: utf-8

"""
Script for the mapping of the various indices and quantities derived from the GPM-IMERG data (for the Island Climate Update "Water Watch")

see also: 

- GPM_update.py (get the list of files to download and update the local dataset)
- GPM_process.py (calculates accumulations, percentiles of scores and number of dry / wet days statistics)

--------------------------------------------------------------------------------------------------------------------------

usage: GPM_map.py [-h] [-d DPATH] [-n NDAYS] [-l LAG] [-ds DPATH_SHAPES] [-o OPATH] [-f FPATH]

optional arguments:

-h, --help              
                        show this help message and exit

-d DPATH, --dpath DPATH
                        
                        the path where to find the netcdf files containing the accumulations and the dry days statistics, REQUIRED

-ds DPATH_SHAPES, --dpath_shapes DPATH_SHAPES
                        
                        the path to the `EEZs` and `Coastlines` folder containing the respective shapefiles, REQUIRED
                        
-f FPATH, --fpath FPATH
                        
                        the path where to save the figures, REQUIRED
--------------------------------------------------------------------------------------------------------------------------
"""

def main(dpath='home/nicolasf/operational/ICU/development/hotspots/outputs/GPM_IMERG', ndays=30, lag=1, dpath_shapes='/home/nicolasf/operational/ICU/development/hotspots/data/shapefiles', fpath='.'): 
    
    import matplotlib

    matplotlib.use('Agg')

    import pathlib
    
    import sys
    
    # import xarray for reading the netcdf files
    
    import xarray as xr

    # import the local modules

    sys.path.append('../../')

    from ICU_Water_Watch import plot, geo, utils
    
    # cast the paths to be pathlib.Paths

    dpath = pathlib.Path(dpath)

    fpath = pathlib.Path(fpath)
    
    dpath_shapes = pathlib.Path(dpath_shapes)

    # get the EEZs 
    
    EEZs, _ = geo.get_EEZs(dpath_shapes=dpath_shapes)

    # get the list of files corresponding to the `ndays` accumulations
    
    lfiles_accumulations = list(dpath.glob(f"GPM_IMERG_accum_{ndays}ndays_to_????-??-??.nc"))
    
    lfiles_accumulations.sort()
    
    lfiles_ndays = list(dpath.glob(f"GPM_IMERG_ndays_{ndays}ndays_to_????-??-??.nc"))

    lfiles_ndays.sort()

    # get the last files 
    
    dset_accum  = xr.open_dataset(lfiles_accumulations[-1])
    
    dset_ndays = xr.open_dataset(lfiles_ndays[-1])

    # accumulation

    plot.map_precip_accum(dset_accum, mask='EEZ', close=False, geoms=EEZs, fpath=fpath)

    # anomalies 

    plot.map_precip_anoms(dset_accum, mask='EEZ', close=False, geoms=EEZs, fpath=fpath)

    # number of dry days 

    plot.map_dry_days_Pacific(dset_ndays, mask='EEZ', geoms=EEZs, fpath=fpath)
    
    # days since last rain 
    
    plot.map_days_since_rain_Pacific(dset_ndays, mask='EEZ', geoms=EEZs, fpath=fpath)
    
    # "Early Action Rainfall" watch definitions

    plot.map_EAR_Watch_Pacific(dset_accum, mask='EEZ', geoms=EEZs, fpath=fpath)
    
    # US Drought Monitor definitions

    plot.map_USDM_Pacific(dset_accum, mask='EEZ', geoms=EEZs, fpath=fpath)
    
    # percentile of score (in decile bins)

    plot.map_decile(dset_accum, mask='EEZ', geoms=EEZs, fpath=fpath)
    
    # interpolate at high resolution 
    
    dset_accum_interp = utils.interp(dset_accum)
    
    coastlines = geo.get_coastlines(dpath_shapes=dpath_shapes)
    
    print("\nnow processing the individual countries / Island groups\n")
    
    for country_name in coastlines.country_na.values: 
    
        print(f"   --> processing {country_name}, EAR Watch then USDM\n")
    
        coastline = coastlines.query(f"country_na == '{country_name}'")
    
        EEZ = EEZs.query(f"COUNTRYNAM == '{country_name}'")
    
        # subset and add EEZ mask 
        dset_sub_country = geo.make_mask_from_gpd(dset_accum_interp, EEZ, subset=True, mask_name='mask_EEZ')
    
        # add country mask 
        dset_sub_country = geo.make_mask_from_gpd(dset_sub_country, coastline, subset=False, mask_name='mask_coastline')
    
        # plots the regional EAR Watch and US 'Drough Monitor' for each country / Island group, with masked EEZ, then masked coastlines
        for mask_type in ['mask_EEZ', 'mask_coastline']: 
            
            # country EAR Watch 
            
            plot.map_EAR_Watch(dset_sub_country, coastlines, coastline, EEZ, mask_name=mask_type, country_name=country_name, fpath=fpath.joinpath('EAR_Watch'), close=True)    
            
            # country US Drought Monitor
            
            plot.map_USDM(dset_sub_country, coastlines, coastline, EEZ, mask_name=mask_type, country_name=country_name, fpath=fpath.joinpath('USDM'), close=True)
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dpath', dest='dpath', type=str, default=None, help='the path where to find the netcdf files containing the accumulations and the dry days statistics, REQUIRED')
    
    parser.add_argument('-n','--ndays', dest='ndays', type=int, default=30, help='the number of days for the above')

    parser.add_argument('-ds','--dpath_shapes', dest='dpath_shapes', type=str, default=None, help='the path to the `EEZs` and `Coastlines` folder containing the respective shapefiles, REQUIRED')
    
    parser.add_argument('-f','--fpath', dest='fpath', type=str, default=None, help='the path where to save the figures, REQUIRED')

    vargs = vars(parser.parse_args())

    # pop out the arguments

    dpath = vargs['dpath']
    ndays = vargs['ndays']
    dpath_shapes = vargs['dpath_shapes']
    fpath = vargs['fpath']

    main(dpath=dpath, ndays=ndays, dpath_shapes=dpath_shapes, fpath=fpath)