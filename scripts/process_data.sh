#!/bin/bash

swot_inroot='../Example/Mile953/Data/SWOT/nc'
swot_outroot='../Example/Mile953/Data/SWOT/geotiff'
emit_inroot='../Example/Mile953/Data/EMIT/nc'
emit_outroot='../Example/Mile953/Data/EMIT/geotiff'
crop_outroot='../Example/Mile953/Data/CROP'


python ~/Github/SpectralHydrograph/SpectralHydrograph/stack_geo.py --swot_inroot $swot_inroot --swot_outroot $swot_outroot --emit_inroot $emit_inroot --emit_outroot $emit_outroot --crop_outroot $crop_outroot
