#!/home/nicolasf/mambaforge/envs/ICU/bin/python
# coding: utf-8

"""
Script for the retrieval of the GPM-IMERG rainfall estimates (for the Island Climate Update "Water Watch")

see also: 

- GPM_process.py (calculates accumulations, percentiles of scores and number of dry / wet days statistics)
- GPM_map.py (maps the various indices and quantities derived from the GPM-IMERG data (for the Island Climate Update "Water Watch"))

--------------------------------------------------------------------------------------------------------------------------

usage: GPM_update.py [-h] [-d DPATH]

optional arguments:

-h, --help              
                        show this help message and exit

-d DPATH, --dpath DPATH
                        
                        the path where to find the GPM-IMERG realtime netcdf files and climatologies, REQUIRED
--------------------------------------------------------------------------------------------------------------------------
"""

def main(dpath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP'): 

    print("now starting assessing the files locally missing\n")

    import pathlib
    
    import sys

    # import the local modules

    sys.path.append('../../')

    from ICU_Water_Watch import GPM
    
    # get the list of files 
    
    lfiles_to_download = GPM.get_files_to_download(dpath)
    
    # now download the files if not empty
    
    if lfiles_to_download is not None: 
        
        GPM.download(dpath=dpath, lfiles=lfiles_to_download) 
        
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-d','--dpath', dest='dpath', type=str, default=None, help='the path where to save the GPM-IMERG realtime netcdf files, REQUIRED')

    vargs = vars(parser.parse_args())

    # pop out the arguments

    dpath = vargs['dpath']

    main(dpath=dpath)