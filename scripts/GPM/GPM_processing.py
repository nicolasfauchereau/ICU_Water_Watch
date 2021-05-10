#!/home/nicolasf/anaconda3/envs/pangeo/bin/python
# coding: utf-8

def main(dpath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP', ndays=30, lag=1, dpath_shapes='/home/nicolasf/operational/ICU/development/hotspots/data/shapefiles', opath='/home/nicolasf/operational/ICU/development/hotspots/outputs/GPM_IMERG', fpath='.'): 
    
    import matplotlib

    matplotlib.use('Agg')

    import pathlib
    
    import sys

    # import the local modules

    sys.path.append('../../')

    from src import plot, GPM, geo, utils

    # update the dataset 
    
    print(f"updating the GPM-IMERG dataset in {str(dpath)}")

    GPM.update(opath=dpath, lag=lag)
    
    # get the list of files corresponding to the number of days passed in the argument `ndays`
    
    print(f"getting the list of files to process for the past {ndays} days")

    lfiles = GPM.get_files_list(dpath = dpath, ndays=ndays, lag=lag)
    
    # make the dataset, and retrieve the attributes

    dset = GPM.make_dataset(lfiles, ndays=ndays)

    last_date, ndays  = GPM.get_attrs(dset)
    
    # calculate the accumulation 
    
    print(f"\ncalculating accumulation for the {ndays} period ending {last_date:%Y-%m-%d}")

    dset_accum = GPM.calculate_realtime_accumulation(dset)

    # get the rain days statistics 
    
    print(f"getting the rain days statistics for the {ndays} period ending {last_date:%Y-%m-%d}")    

    dset_ndays = GPM.get_rain_days_stats(dset)

    # get the climatology, for the calculation of the anomalies 

    clim = GPM.get_climatology(dpath=dpath, ndays=ndays, date=last_date)

    # calculate the anomalies and percentile of scores, from the accumulation dataset

    dset_accum = GPM.calc_anoms_and_pctscores(dset_accum, clim)

    # get the EEZs for each country, and a dissolved version for the mask 

    EEZs, merged_EEZs = geo.get_EEZs(dpath_shapes=dpath_shapes)
    
    # adds the mask to the datasets

    dset_accum = geo.make_mask_from_gpd(dset_accum, merged_EEZs, subset=True, mask_name='EEZ')

    dset_ndays = geo.make_mask_from_gpd(dset_ndays, merged_EEZs, subset=True, mask_name='EEZ')
    
    # saves the Regional Pacific files to disk 
    
    GPM.save(dset_accum, opath=opath, kind='accum')
        
    GPM.save(dset_ndays, opath=opath, kind='ndays')
    
    # NOW start the plotting

    plot.map_precip_anoms(dset_accum, mask='EEZ', close=False, geoms=EEZs, fpath=fpath)

    plot.map_dry_days_Pacific(dset_ndays, mask='EEZ', geoms=EEZs, close=False)

    plot.map_EAR_Watch_Pacific(dset_accum, mask='EEZ', geoms=EEZs, close=False, fpath=fpath)

    plot.map_USDM_Pacific(dset_accum, mask='EEZ', geoms=EEZs, close=False, fpath=fpath)

    plot.map_decile(dset_accum, mask='EEZ', geoms=EEZs, close=False, fpath=fpath)
    
    # interpolate at high resolution 
    
    dset_accum_interp = utils.interp(dset_accum)
    
    coastlines = geo.get_coastlines(dpath_shapes=dpath_shapes)
    
    print("\nnow processing the individual countries / Island groups")
    
    dpath = pathlib.Path(dpath)
    
    fpath = pathlib.Path(fpath)
    
    for country_name in coastlines.country_na.values: 
    
        print(f"\n  --> processing {country_name}, EAR Watch then USDM")
    
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

    parser.add_argument('-d','--dpath', dest='dpath', type=str, default=None, help='the path where to find the GPM-IMERG realtime netcdf files and climatologies, REQUIRED')
    
    parser.add_argument('-n','--ndays', dest='ndays', type=int, default=30, help='the number of days over which to calculate the accumulation and take the climatology')

    parser.add_argument('-l','--lag', dest='lag', type=int, default=1, help='the lag (in days) to realtime, if run in the morning (NZ time) 1 should be OK')

    parser.add_argument('-ds','--dpath_shapes', dest='dpath_shapes', type=str, default=None, help='the path to the `EEZs` and `Coastlines` folder containing the respective shapefiles, REQUIRED')

    parser.add_argument('-o','--opath', dest='opath', type=str, default=None, help='the path where to save the outputs files (netcdf and geotiff), REQUIRED')
    
    parser.add_argument('-f','--fpath', dest='fpath', type=str, default=None, help='the path where to save the figures, REQUIRED')

    vargs = vars(parser.parse_args())

    # pop out the arguments

    dpath = vargs['dpath']
    ndays = vargs['ndays']
    lag = vargs['lag']
    dpath_shapes = vargs['dpath_shapes']
    opath = vargs['opath']
    fpath = vargs['fpath']

    main(dpath=dpath, ndays=ndays, lag=lag, dpath_shapes=dpath_shapes, opath=opath, fpath=fpath)