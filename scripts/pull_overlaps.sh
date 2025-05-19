#!/bin/bash


emit_path='../Example/Mile953/Shapes/emit_data.gpkg'
swot_path='../Example/Mile953/Shapes/swot_data.gpkg'
out_root='../Example/Mile953/Overlaps'


python ../SpectralHydrograph/overlapping_shapes.py $emit_path $swot_path $out_root
