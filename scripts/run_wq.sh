#!/bin/bash
path='../Example/Mile953/Data/CSV/EMIT_L2A_RFL_001_20230203T171434_sample.csv'
out_root='../Example/Mile953/Data/CSV'
n_cores=15


python /Users/bgreenbe/Projects/SWOT/run_wq.py $path $out_root --n_cores $n_cores
