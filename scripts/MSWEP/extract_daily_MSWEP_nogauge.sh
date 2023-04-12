#!/usr/bin/bash 

# extract a domain from the daily no gauge dataset
# and compress, see compress_netcdf.py

lon_min=100
lon_max=240
lat_min=-50
lat_max=30

cdo='/home/nicolasf/mambaforge/envs/CDO/bin/cdo'
indir='/media/nicolasf/END19101/ICU/data/glo2ho/MSWEP280/Past_nogauge/Daily'
outdir='/media/nicolasf/END19101/ICU/data/MSWEP/Daily/subsets_nogauge/SP'
compress_script='/home/nicolasf/operational/ICU/development/hotspots/code/ICU_Water_Watch/scripts/MSWEP/compress_netcdf.py'

cd ${indir};

for f in *.nc; do ${cdo} sellonlatbox,${lon_min},${lon_max},${lat_min},${lat_max} ${f} ${outdir}/${f}; ${compress_script} ${outdir}/${f}; done