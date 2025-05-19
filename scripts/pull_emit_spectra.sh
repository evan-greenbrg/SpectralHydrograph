#!/bin/bash

emit_path='../Example/Mile953/Data/CROP/EMIT/RFL/EMIT_L2A_RFL_001_20230203T171434_2303412_006_rfl_epsg_32615_epsg_32615_CROP.tif'
water_path='../Example/Mile953/Data/CROP/EMIT/WATER/EMIT_L2A_RFL_001_20241002T165450_2427611_019_rfl_epsg_32615_epsg_32615_CROP_WATER.tif'
out_root='../Example/Mile953/Data/CSV'
roi_path='../Example/Mile953.gpkg'

python /Users/bgreenbe/Projects/SWOT/pull_emit_spectra.py $emit_path $water_path $roi_path $out_root
