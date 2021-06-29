#!/usr/bin/bash

dpath='/home/nicolasf/operational/ICU/ops/data/GPM_IMERG/daily/extended_SP'
dpath_shapes='/home/nicolasf/operational/ICU/development/hotspots/data/shapefiles'
opath='/home/nicolasf/operational/ICU/development/hotspots/outputs/GPM_IMERG'
figures_path='/home/nicolasf/operational/ICU/development/hotspots/figures/GPM_IMERG'
lag=2

# --------------------------------------------------------------------------------------
# uncomment this to generate `retrospective` outputs and maps (over a sequence of lags)

# for lag in $(seq 1 30); do 
# 	echo ${lag}; 
# 	for ndays in 30 60 90 180 360; do 
# 	 	./GPM_processing.py --dpath=${dpath} --ndays=${ndays} --lag=${lag} --dpath_shapes=${dpath_shapes} --opath=${opath} --fpath=${figures_path}; 
# 	done;
# done; 


for ndays in 30 60 90 180 360; do
	./GPM_processing.py --dpath=${dpath} --ndays=${ndays} --lag=${lag} --dpath_shapes=${dpath_shapes} --opath=${opath} --fpath=${figures_path};
done; 
