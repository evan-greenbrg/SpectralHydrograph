#!/bin/bash
swot_path='../Example/Mile953/Data/CROP/SWOT/SWOT_wse_stack_epsg_32616_CROP.tif'
water_path='../Example/Mile953/Data/CROP/EMIT/WATER/EMIT_L2A_RFL_001_20241002T165450_2427611_019_rfl_epsg_32615_epsg_32615_CROP_WATER.tif'
roi_path='../Example/Mile953.gpkg'
swot_water_out='../Example/Mile953/Data/CROP/SWOT/WATER/SWOT_WATER_epsg_32615.tif'
data_out='../Example/Mile953/Data/CROP/SWOT/csv/SWOT_wse_data.csv'

python /Users/bgreenbe/Projects/SWOT/pull_wse_data.py $swot_path $water_path $roi_path --swot_water_out $swot_water_out --data_out $data_out
