#!/bin/bash

poly_path='../Example//Mile953.gpkg'
start_date='2023-01-01'
end_date='2026-01-01'
cloud=10
outroot='../Example/Mile953'

python ../SpectralHydrograph/pull_shapes.py $poly_path --start_date $start_date --end_date $end_date --max_cloud $cloud --out_root $outroot
