#!/home/nicolasf/mambaforge/envs/ICU/bin/python
# coding: utf-8

"""
Script for the processing of the GPM-IMERG rainfall estimates (for the Island Climate Update "Water Watch")

see also: 

- GPM_update.py (get the list of files to download and update the local dataset)
- GPM_map.py (maps the various indices and quantities derived from the GPM-IMERG data (for the Island Climate Update "Water Watch"))

--------------------------------------------------------------------------------------------------------------------------

usage: GPM_process.py [-h] [-d DPATH] [-n NDAYS] [-l LAG] [-ds DPATH_SHAPES] [-o OPATH] [-f FPATH]

optional arguments:

-h, --help              
                        show this help message and exit

-d DPATH, --dpath DPATH
                        
                        the path where to find the GPM-IMERG realtime netcdf files and climatologies, REQUIRED

-n NDAYS, --ndays NDAYS
                        
                        the number of days over which to calculate the accumulation and take the climatology

-l LAG, --lag LAG       
                        
                        the lag (in days) to realtime, default to 2 days to realtime given the latency and time difference between NZ time and UTC

-ds DPATH_SHAPES, --dpath_shapes DPATH_SHAPES
                        
                        the path to the `EEZs` and `Coastlines` folder containing the respective shapefiles, REQUIRED

-o OPATH, --opath OPATH
                        
                        the path where to save the outputs files (netcdf and geotiff), REQUIRED
--------------------------------------------------------------------------------------------------------------------------
"""

def main(dpath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP', ndays=30, lag=2, dpath_shapes='/home/nicolasf/operational/ICU/development/hotspots/data/shapefiles', opath='/home/nicolasf/operational/ICU/development/hotspots/outputs/GPM_IMERG'): 

    import pathlib
    
    import sys

    # import the local modules

    sys.path.append('../../')

    from ICU_Water_Watch import GPM, geo 
    
    print(f"getting the list of files to process for the past {ndays} days\n")

    lfiles = GPM.get_files_list(dpath = dpath, ndays=ndays, lag=lag)
    
    filenames = [f.name for f in lfiles]
    
    print(f"list of files is of length {len(lfiles)}\n")
    
    print("\n".join(filenames))
    
    # make the dataset, and retrieve the attributes

    dset = GPM.make_dataset(lfiles, ndays=ndays)

    last_date, ndays  = GPM.get_attrs(dset)
    
    # calculate the accumulation 
    
    print(f"calculating accumulation for the {ndays} days period ending {last_date:%Y-%m-%d}\n")

    dset_accum = GPM.calculate_realtime_accumulation(dset)

    # get the rain days statistics 
    
    print(f"getting the rain days statistics for the {ndays} days period ending {last_date:%Y-%m-%d}\n")    

    dset_ndays = GPM.get_rain_days_stats(dset)

    # get the climatology, for the calculation of the anomalies 

    clim = GPM.get_climatology(dpath=dpath, ndays=ndays, date=last_date)

    # calculate the anomalies and percentile of scores, from the accumulation dataset

    dset_accum = GPM.calc_anoms_and_pctscores(dset_accum, clim)

    # get a dissolved version of the EEZs for the masks 

    _, merged_EEZs = geo.get_EEZs(dpath_shapes=dpath_shapes)
    
    # adds the mask to the datasets

    dset_accum = geo.make_mask_from_gpd(dset_accum, merged_EEZs, subset=True, mask_name='EEZ')

    dset_ndays = geo.make_mask_from_gpd(dset_ndays, merged_EEZs, subset=True, mask_name='EEZ')
    
    # saves the Regional Pacific files to disk 
    
    GPM.save(dset_accum, opath=opath, kind='accum')
        
    GPM.save(dset_ndays, opath=opath, kind='ndays')
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dpath', dest='dpath', type=str, default=None, help='the path where to find the GPM-IMERG realtime netcdf files and climatologies, REQUIRED')
    
    parser.add_argument('-n','--ndays', dest='ndays', type=int, default=30, help='the number of days over which to calculate the accumulation and take the climatology')

    parser.add_argument('-l','--lag', dest='lag', type=int, default=2, help='the lag (in days) to realtime, default to 2 given the latency and the time difference between NZ time and UTC')

    parser.add_argument('-ds','--dpath_shapes', dest='dpath_shapes', type=str, default=None, help='the path to the `EEZs` and `Coastlines` folder containing the respective shapefiles, REQUIRED')

    parser.add_argument('-o','--opath', dest='opath', type=str, default=None, help='the path where to save the outputs files (netcdf and geotiff), REQUIRED')
    
    vargs = vars(parser.parse_args())

    # pop out the arguments

    dpath = vargs['dpath']
    ndays = vargs['ndays']
    lag = vargs['lag']
    dpath_shapes = vargs['dpath_shapes']
    opath = vargs['opath']
    
    main(dpath=dpath, ndays=ndays, lag=lag, dpath_shapes=dpath_shapes, opath=opath)