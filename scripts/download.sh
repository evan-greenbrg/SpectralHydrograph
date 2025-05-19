#!/bin/bash

path='../Example/Mile953/Overlaps/emit_swot_overlap_1_images.gpkg'
out_root='../Example/Mile953/Data'
swot_code='45F'


python ../SpectralHydrograph/download_data.py $path $out_root
