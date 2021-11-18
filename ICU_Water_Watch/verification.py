"""
The `verification` module contains helper functions for the verification of deterministic and probabilistic 
C3S forecasts against observations or reanalyses 
"""

import pathlib
import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime, date 

from . import utils
from . import GPM 
from . import C3S 
from . import geo
from . import domains 

def get_era5(dpath='/media/nicolasf/END19101/data/REANALYSIS/downloads', varname='precip', start=1993, end=2016, month_begin=True, chunks=None, interp=True):  

    if not(type(dpath) == pathlib.PosixPath): 
        dpath = pathlib.Path(dpath)
        
    dpath = dpath.joinpath(varname.upper())
    
    lfiles = list(dpath.glob("*.nc"))
    
    lfiles.sort()
    
    lfiles = [x for x in lfiles if (int(x.name.split('_')[-2]) >= start) and (int(x.name.split('_')[-2]) <= end)]

    dset = xr.open_mfdataset(lfiles, parallel=True, concat_dim='time', combine='nested', preprocess=preprocess_era)
    
    if varname == 'precip': 
        
        dset['precip'] = dset['precip'] * 1000.
        dset['precip'].attrs['unit'] = 'mm'
    
    if month_begin and np.alltrue(dset['time'].dt.day != 1):
        
        dset['time'] = dset['time'].to_index() - pd.offsets.MonthBegin()
        
    if interp: 
        
        dset = utils.interp_to_1x(dset)
    
    clim = dset.groupby(dset.time.dt.month).mean('time')
    
    dset_anomalies = dset.groupby(dset.time.dt.month) - clim
    
    return dset, dset_anomalies
    
def preprocess_era(dset): 
    
    dvars = {}
    dvars['tp'] = 'precip'
    
    for k in dvars.keys(): 
        if k in dset.data_vars: 
            dset = dset.rename({k:dvars[k]})
    
    dset = dset.rename({'latitude':'lat', 'longitude':'lon'})
    
    dset = dset.sortby('lat')
    
    # if the longitude 
    
    if dset['lon'][0] < 0: 
        
        dset = utils.roll_longitudes(dset)
    
    return dset

def get_mswep(dpath='/media/nicolasf/END19101/data/MSWEP/data/global_monthly_1deg/', start=1993, end=2016, month_begin=True, chunks=None):
    
    if not(type(dpath) == pathlib.PosixPath): 
        dpath = pathlib.Path(dpath)
        
    lfiles = list(dpath.glob("*.nc"))
    
    lfiles.sort()
    
    start = (start * 100) + 1
    end = (end * 100) + 12
    
    lfiles = [x for x in lfiles if (int(x.name.split('.')[0]) >= start) and (int(x.name.split('.')[0]) <= end)]
    
    if chunks is not None: 
        
        dset = xr.open_mfdataset(lfiles, parallel=True, concat_dim='time', chunks=chunks)
        
    else: 
    
        dset = xr.open_mfdataset(lfiles, parallel=True, concat_dim='time')
    
    if month_begin and np.alltrue(dset['time'].dt.day != 1):
        
        dset['time'] = dset['time'].to_index() - pd.offsets.MonthBegin()
    
    clim = dset.groupby(dset.time.dt.month).mean('time')
    
    dset_anomalies = dset.groupby(dset.time.dt.month) - clim
    
    return dset, dset_anomalies

def get_cmap(dpath='/media/nicolasf/END19101/data/CMAP', fname='precip.mon.mean.nc', start=1993, end=2016, month_begin=True, accum=True, domain=None, chunks=None): 
    """
    [summary]

    [extended_summary]

    Parameters
    ----------
    dpath : str, optional
        [description], by default '/media/nicolasf/END19101/data/CMAP'
    fname : str, optional
        [description], by default 'precip.mon.mean.nc'
    start : int, optional
        [description], by default 1993
    end : int, optional
        [description], by default 2016
    month_begin : bool, optional
        [description], by default True
    accum : bool, optional
        [description], by default True
    domain : [type], optional
        [description], by default None
    chunks : [type], optional
        [description], by default None

    Returns
    -------
    tuple
        (xarray.Dataset of raw values, xarray.Dataset of anomalies)
    """
    from datetime import datetime
    
    if not(type(dpath) == pathlib.PosixPath): 
        dpath = pathlib.Path(dpath)
        
    if chunks is not None:
        
        dset = xr.open_dataset(dpath.joinpath(fname), chunks=chunks)
    
    else: 
        
        dset = xr.open_dataset(dpath.joinpath(fname))
    
    dset = dset.sel(time=slice(f"{start}-01-01",  f"{end}-12-31"))
    
    dset = dset.sortby('lat')
    
    if accum: 
        
        dset = utils.convert_rainfall_OBS(dset, varin='precip')
        
        dset = dset.drop('ndays')
        
    if month_begin and np.alltrue(dset['time'].dt.day != 1):
        
        dset['time'] = dset['time'].to_index() - pd.offsets.MonthBegin()
        
    if domain is not None: 
        
        dset = dset.sel(lon=slice(*domain[:2]), lat=slice(*domain[2:]))
        
    # calculates the climatology 
        
    clim = dset.groupby(dset.time.dt.month).mean('time')
    
    dset_anomalies = dset.groupby(dset.time.dt.month) - clim
    
    return dset, dset_anomalies

def get_gpcp(dpath='/media/nicolasf/END19101/data/GPCP', fname='precip.mon.mean.nc', start=1993, end=2016, month_begin=True, accum=True, domain=None, chunks=None): 
        
    from datetime import datetime
    
    if not(type(dpath) == pathlib.PosixPath): 
        dpath = pathlib.Path(dpath)
        
    if chunks is not None:
        
        dset = xr.open_dataset(dpath.joinpath(fname), chunks=chunks)
    
    else: 
        
        dset = xr.open_dataset(dpath.joinpath(fname))
    
    dset = dset.sel(time=slice(f"{start}-01-01",  f"{end}-12-31"))
    
    dset = dset.sortby('lat')
    
    if accum: 
        
        dset = utils.convert_rainfall_OBS(dset, varin='precip')
        
        dset = dset.drop('ndays')
        
    if month_begin and np.alltrue(dset['time'].dt.day != 1):
        
        dset['time'] = dset['time'].to_index() - pd.offsets.MonthBegin()
        
    if domain is not None: 
        
        dset = dset.sel(lon=slice(*domain[:2]), lat=slice(*domain[2:]))
        
    # calculates the climatology 
        
    clim = dset.groupby(dset.time.dt.month).mean('time')
    
    dset_anomalies = dset.groupby(dset.time.dt.month) - clim
    
    return dset, dset_anomalies

def get_OISST(dpath='/media/nicolasf/END19101/ICU/data/SST/NOAA_OISSTv2/monthly/', filename='OISST_v2_monthly.nc', start=1993, end=2016, domain=None, detrend=False): 
    
    dpath = pathlib.Path(dpath).joinpath(filename)

    dset = xr.open_dataset(dpath)
    
    dset = dset.sel(time=slice(str(start), str(end)))

    if domain is not None: 
        
        dset = dset.sel(lon=slice(*domain[:2]), lat=slice(*domain[2:]))

    dset_anoms = dset.groupby('time.month') - dset.groupby('time.month').mean('time') 
        
        # detrend 
        
    if detrend: 
            
        dset_anoms['sst']  = utils.detrend_dim(dset_anoms['sst'], 'time')

    # compute the anomalies 
    
    dset = dset.compute()
    dset_anoms = dset_anoms.compute()
        
    if 'month' in dset_anoms.coords: 
        dset_anoms = dset_anoms.drop('month')

    return dset, dset_anoms

def get_ERSST(access='opendap', start=1993, end=2016, month_begin=True, domain=None, chunks=None, detrend=False): 
    """

    get the ERSST dataset and calculates anomalies 

    Parameters
    ----------
    access : str, optional
        The access method, can be 
        - 'opendap' (default): access http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc
        - 'google_cloud': access at gs://pangeo-noaa-ncei/noaa.ersst.v5.zarr (requires gcsfs library)
        - or any string that points to a folder containing the individual monthly files, with pattern ersst.YYYYMM.nc
        typically downloaded from ftp://ftp.ncdc.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ 
        
        default is 'opendap'
    start : int, optional
        [description], by default 1993
    end : int, optional
        [description], by default 2016
    month_begin : bool, optional
        [description], by default True
    domain : [type], optional
        [description], by default None
    chunks : [type], optional
        [description], by default None
    """
    
    if access == 'opendap': 
        
        url = "http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc"
        
        print(f"reading monthly ERSST dataset from {url}")
        
        ersst = xr.open_dataset(url, drop_variables=["time_bnds"]) 
        
        if domain is not None: 
        
            ersst = ersst.sel(time=slice(str(start), str(end)), lon=slice(*domain[:2]), lat=slice(*domain[2:]))
        
        else:

            ersst = ersst.sel(time=slice(str(start), str(end)))
        
        # calculates anomalies 
        
        ersst_anoms = ersst.groupby('time.month') - ersst.groupby('time.month').mean('time') 
        
        # detrend 
        
        if detrend: 
            
            ersst_anoms['sst']  = utils.detrend_dim(ersst_anoms['sst'], 'time')
        
        ersst = ersst.compute()
        ersst_anoms = ersst_anoms.compute()
        
        if 'month' in ersst_anoms.coords: 
            ersst_anoms = ersst_anoms.drop('month')
        
        return ersst, ersst_anoms
            
    elif access == 'google_cloud': 
        
        try:

            import gcsfs 

            fs = gcsfs.GCSFileSystem(token="anon")

            ersst = xr.open_zarr(fs.get_mapper("gs://pangeo-noaa-ncei/noaa.ersst.v5.zarr"), consolidated=True).load()
        
            if domain is not None: 

                ersst = ersst.sel(time=slice(str(start), str(end)), lon=slice(*domain[:2]), lat=slice(*domain[2:]))

            else:

                ersst = ersst.sel(time=slice(str(start), str(end)))                 

            ersst_anoms = ersst.groupby('time.month') - ersst.groupby('time.month').mean('time')
                
            if detrend: 
            
                ersst_anoms['sst']  = utils.detrend_dim(ersst_anoms['sst'], 'time')
        
            ersst = ersst.compute() 
            ersst_anoms = ersst_anoms.compute()
            
            if 'month' in ersst_anoms.coords: 
                ersst_anoms = ersst_anoms.drop('month')
            
            return ersst, ersst_anoms
        
        except: 
            
            raise Exception(f"not possible to use google cloud")
            
            return None
    
    else: 
        
        dpath = pathlib.Path(access)
        
        lfiles = list(dpath.glob("ersst.??????.nc")) 
        
        lfiles.sort() 
        
        # filter the list of files 
        
        lfiles = [x for x in lfiles if ((int(x.name.split('.')[1]) // 100) >= start) and ((int(x.name.split('.')[1]) // 100) <= end)]
        
        ersst = xr.open_mfdataset(lfiles, concat_dim='time', parallel=True, drop_variables=["ssta"]).squeeze()
        
        if domain is not None:
            
            ersst = ersst.sel(lon=slice(*domain[:2]), lat=slice(*domain[2:]))
            
        ersst_anoms = ersst.groupby('time.month') - ersst.groupby('time.month').mean('time') 
        
        if 'month' in ersst_anoms.coords: 
            ersst_anoms = ersst_anoms.drop('month')
        
        if detrend: 
            
            ersst_anoms['sst']  = utils.detrend_dim(ersst_anoms['sst'], 'time')
        
        ersst = ersst.compute()
        ersst_anoms = ersst_anoms.compute() 
        
        # fix the calendar (was '360_day')
        
        start_date = ersst.time.to_index()[0]
        end_date = ersst.time.to_index()[-1]
        
        start_date = date(start_date.year, start_date.month, start_date.day)
        end_date = date(end_date.year, end_date.month, end_date.day)
        
        ersst['time'] = pd.date_range(start=start_date, end=end_date, freq='MS')
        ersst_anoms['time'] = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        return ersst, ersst_anoms

def process_for_climpred(dset, time_name='time', lead_name='step'): 
    """
    rename the 'time' to 'init and 'step' to 'lead'
    as well as add attribute "months" to the 'lead'
    variable to make it compatible with climpred

    Parameters
    ----------
    dset : xarray.Dataset
        input dataset 
    time_name : str, optional
        name of the time variable corresponding to initial time, by default 'time'
    lead_name : str, optional
        name of the leadtime variable (integer), by default 'step'

    Returns
    -------
    xarray.Dataset
    """
    dset = dset.rename({time_name:'init',lead_name:'lead'})
    dset['lead'].attrs = {'units':'months'}
    return dset 

def make_mask_from_EOF(eofs, mode=0, threshold=0.7, verif_dset='MSWEP'): 
    """
    make a mask (np.nan or 1) from a dataset of eofs (see validation/make_obs_PCA.ipynb)

    Parameters
    ----------
    eofs : xr.Dataset
        The dataset containing the EOFs
    mode : int, optional
        The EOF number, by default 0
    threshold : float, optional
        The correlation threshold, by default 0.7
    verif_dset : str, optional
        The name of the verification dataset, by default 'MSWEP'
        can be either 'MSWEP' or 'CMAP'

    Returns
    -------
    xarray.DataArray 
        The mask 
    """
    
    eof = eofs.sel(mode=mode)
    
    if np.sign(threshold) == -1.: 
        eof = eof.where(eof <= threshold)[f'EOFs_{verif_dset}']
    elif np.sign(threshold) == 1.:
        eof = eof.where(eof >= threshold)[f'EOFs_{verif_dset}']
    
    mask_eof = eof.where(np.isnan(eof), 1)
    
    return mask_eof

#######################################################
def Nino(dset, lon='lon', lat='lat', time='time', avg=5, nino='3.4'):
	"""
		Produce ENSO index timeseries from dset according to Technical Notes
		 guidance from UCAR: https://climatedataguide.ucar.edu/climate-data/nino-dset-indices-nino-12-3-34-4-oni-and-tni
		INPUTS:
			dset:  xarray.DataArray which will be averaged over Nino domains
			lon:  name of longitude dimension. Has to be in [0,360].
			lat:  name of latitude dimension. Has to be increasing.
			time: name of time dimension.
			avg:  size of rolling window for rolling time average.
			nino: which Nino index to compute. Choices are
					'1+2','3','4','3.4','oni','tni'
		OUTPUTS:
			dset: spatially averaged over respective Nino index domain
				  note that no running means are performed.
	"""
	ninos = {
		'1+2' : {lon:slice(270,280),lat:slice(-10,0)},
		'3'   : {lon:slice(210,270),lat:slice(-5,5)},
		'4'   : {lon:slice(160,210),lat:slice(-5,5)},
		'3.4' : {lon:slice(190,240),lat:slice(-5,5)},
		'oni' : {lon:slice(190,240),lat:slice(-5,5)},
	}
	possible_ninos = list(ninos.keys())+['tni']
	if nino not in possible_ninos:
		raise ValueError('Nino type {0} not recognised. Possible choices are {1}'.format(nino,', '.join(possible_ninos)))
	lon_name = None
	lat_name = None
	if dset[lon].min() < 0 or dset[lon].max() <= 180:
		lon_name = lon
	if dset[lat][0] > dset[lat][-1]:
		lat_name = lat
	if lon_name is not None or lat_name is not None:
		print('WARNING: re-arranging dset to be in domain [0,360] x [-90,90]')
		dset = utils.roll_longitudes(dset)

	def NinoAvg(dset,nino,time,avg):
		dseta = dset.sel(ninos[nino]).mean(dim=[lon,lat])
		dsetc = dseta.groupby('.'.join([time,'month'])).mean(dim=time)
		dseta = dseta.groupby('.'.join([time,'month'])) - dsetc
		if avg is not None:
			dseta = dseta.rolling({time:avg}).mean()
		return dseta/dseta.std(dim=time)

	if nino == 'tni':
		n12 = NinoAvg(dset,'1+2',time,None)
		n4  = NinoAvg(dset,'4',time,None)
		tni = (n12-n4).rolling({time:avg}).mean()
		return tni/tni.std(dim=time)
	else:
		return NinoAvg(dset,nino,time,avg)

def get_CPC_NINO(sst='oisst', region='3.4', clim=[1993,2016]): 
    """
    """
    
    if sst == 'oisst': 
        url = 'https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices'
    elif sst == 'ersst': 
        url = 'http://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii'
    
    import numpy as np 
    import pandas as pd 
    from datetime import datetime
    
    nino = pd.read_csv(url, sep='\s+', engine='python')
    
    nino = nino[['YR','MON',f'NINO{region}']]
    
    nino.loc[:,'DAY'] = 1
    
    nino_clim = nino.copy()
    nino_clim.index = nino_clim.YR
    nino_clim = nino_clim.loc[clim[0]:clim[1],:]
    nino_clim = nino_clim.groupby(nino_clim.MON).mean()
    
    nino.index = nino[['YR', 'MON', 'DAY']].apply(lambda d : datetime(*d), axis = 1)
    
    def demean(x): 
        return x - x.loc[str(clim[0]):str(clim[1])].mean()
    
    nino['anoms'] = nino.groupby(nino.MON)[[f'NINO{region}']].transform(demean)
    
    ninos = nino[['anoms']]
    
    ninos.columns = [f'NINO{region}']
    
    return ninos

def enso_to_xarray(enso): 
    enso = enso.to_xarray()
    enso = enso.rename({'index':'init'})
    return enso

def make_ENSO_ACC_RMSE(MME, dset_obs, enso_index, GCM='MME', name='El Nino', plot_map=True, fname=None, cmap=None): 

    import numpy as np 
    import xarray as xr 
    import cartopy.crs as ccrs 
    from climpred import HindcastEnsemble
    from dask.diagnostics import ProgressBar
    from matplotlib import pyplot as plt
    
    if type(enso_index) != xr.core.dataset.Dataset: 
        
        enso_index = enso_to_xarray(enso_index)
    
    MME_enso = MME.sel(init=enso_index.init)
    
    dset_gcm = MME_enso.sel(GCM=GCM)
    
    dset_gcm = dset_gcm.dropna(dim='init')
    
    hindcast = HindcastEnsemble(dset_gcm)
    
    hindcast = hindcast.add_observations(dset_obs)
    
    R_map = hindcast.verify(metric='pearson_r', comparison='e2o', dim=['init'], alignment='maximize')
    
    if plot_map: 
    
        if cmap is None: 
            
            cmap = plt.cm.Greens
    
        fg = R_map['precip'].plot.contourf(col='lead', levels=np.arange(0, 1.1, 0.1), cmap=cmap, \
                                                  subplot_kws={'projection':ccrs.PlateCarree(central_longitude=180)}, 
                                                 transform=ccrs.PlateCarree(), cbar_kwargs={'shrink':0.5, 'orientation':'horizontal', 'aspect':50, 'label':"Pearson's R"}) 


        [ax.coastlines() for ax in fg.axes.flat]

        axes = fg.axes.flat

        [axes[i].set_title(f"R, {GCM}: {i + 1} months lead, {name} init") for i in range(len(axes))]

        if fname is not None: 
        
            fg.fig.savefig(fname, dpi=200, bbox_inches='tight',facecolor='w')
            
        else: 
            
            fg.fig.savefig(f"./R_maps_{GCM}_{name.replace(' ','_')}.png", dpi=200, bbox_inches='tight',facecolor='w')
            
        ACC = hindcast.verify(metric='pearson_r', comparison='e2o', dim=['init','lat','lon'], alignment='maximize')

        RMSE = hindcast.verify(metric='rmse', comparison='e2o', dim=['init','lat','lon'], alignment='maximize')
        
        with ProgressBar(): 
            ACC = ACC.compute()
            RMSE = RMSE.compute()
            
    return R_map, ACC, RMSE
