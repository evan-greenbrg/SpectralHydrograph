#!/bin/bash
path='../Example/Mile953/Data/CROP/EMIT/EMIT_L2A_RFL_001_20240413T201909_2410413_036_rfl_epsg_32616_epsg_32616_CROP.tif'
out_path='../Example/Mile953/Data/CROP/EMIT/WATER/EMIT_L2A_RFL_001_20240413T201909_2410413_036_rfl_epsg_32616_epsg_32616_CROP_WATER.tif'

echo $path
python /Users/bgreenbe/Github/SpectralHydrograph/SpectralHydrograph/quantify_water.py $path $out_path
