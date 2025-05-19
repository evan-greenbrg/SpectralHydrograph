import glob
import re
import os
import datetime
import click
import earthaccess
import numpy as np
import pandas
import geopandas
import xarray
import json
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from osgeo import gdal
import rasterio
import rioxarray
import pyproj
from rasterio.transform import Affine
from rasterio import merge
from shapely import geometry
from shapely.ops import transform
import fiona



def convert_swot_to_geotiff(inroot, outroot, variable):
    os.makedirs(os.path.join(outroot, variable), exist_ok=True)
    fps = glob.glob(os.path.join(inroot, '*.nc'))
    for fp in fps:
        print(fp)
        xds = xarray.open_dataset(fp)
        epsg = pyproj.CRS(xds.crs.crs_wkt).to_epsg()

        xds = xds.rio.write_crs(f"epsg:{epsg}", inplace=True)

        # Parse name
        name = fp.split('/')[-1]
        out_name = name.split('.')
        out_name = out_name[0] + f'_{variable}.tif'
        out_path = os.path.join(outroot, variable, out_name)
        xds[variable].rio.to_raster(out_path)


def convert_emit_to_geotiff(inroot, outroot):
    os.makedirs(os.path.join(outroot), exist_ok=True)

    fps = glob.glob(os.path.join(inroot, '*.nc'))

    for fp in fps:
        print(fp)
        ds = Dataset(fp)
        crs = pyproj.CRS(ds.spatial_ref)
        geotransform = Affine.from_gdal(*ds.geotransform)
        wl = np.array(ds.groups['sensor_band_parameters']['wavelengths'])
        use_wl = np.squeeze(np.argwhere((wl >= 400) & (wl <=900)))

        glt_x = np.array(ds.groups['location']['glt_x'])
        glt_y = np.array(ds.groups['location']['glt_y'])
        glt_idx = np.argwhere(glt_x)

        rfl = np.array(ds['reflectance'])
        rfl_glt = np.zeros((glt_x.shape[0], glt_x.shape[1], rfl.shape[2])) + -9999.

        vals = rfl[
            glt_y[glt_idx[:, 0], glt_idx[:, 1]] - 1,
            glt_x[glt_idx[:, 0], glt_idx[:, 1]] - 1,
            :
        ]
        rfl_glt[glt_idx[:, 0], glt_idx[:, 1], :] = vals
        rfl_glt = rfl_glt[..., use_wl]

        out_meta = {
            'driver':'GTiff',
            'width': rfl_glt.shape[1],
            'height': rfl_glt.shape[0],
            'count': rfl_glt.shape[2],
            'dtype':'float64',
            'crs': crs,
            'transform': geotransform,
            'nodata':-9999.
        }
        name = fp.split('/')[-1]
        out_name = name.split('.')
        out_name = out_name[0] + f'_rfl.tif'
        out_path = os.path.join(outroot, out_name)
        with rasterio.open(fp=out_path, mode='w', **out_meta) as dst:
            dst.descriptions = [str(w) for w in wl[use_wl]]
            dst.write(np.moveaxis(rfl_glt, 2, 0))

    return wl[use_wl]


def reproject(emit_ds, emit_crs, swot_crs, out_path):
    transform, width, height = rasterio.warp.calculate_default_transform(
        emit_crs, 
        swot_crs, 
        emit_ds.width, 
        emit_ds.height, 
        *emit_ds.bounds
    )
    kwargs = emit_ds.meta.copy()
    kwargs.update({
        'crs': swot_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    with rasterio.open(out_path, 'w', **kwargs) as dst:
        for i in range(1, emit_ds.count + 1):
            rasterio.warp.reproject(
                rasterio.band(emit_ds, i),
                rasterio.band(dst, i),
                src_transform=emit_ds.transform,
                src_crs=emit_ds.crs,
                dst_transform=transform,
                dst_crs=swot_crs,
                resampling=rasterio.warp.Resampling.nearest,
                src_nodata=-9999.,
                dst_nodata=-9999.
            )

    return rasterio.open(out_path)


def upright_image(ds, fp):
    tf = list(ds.transform)
    if tf[4] > 0:
        ydist = ds.shape[1] * tf[4]
        xdist = ds.shape[0] * tf[0]
        im = np.flip(ds.read(), axis=1)
        tf[4] = -tf[4]
        # tf[2] -= xdist
        tf[5] += ydist
    else:
        return ds

    meta = ds.meta
    meta['transform'] = Affine(*tf)

    with rasterio.open(fp=fp, mode='w', **meta) as dst:
        dst.write(im)

    return rasterio.open(fp)


def stack_arrays(ds_list: list, crs, out_root):
    mosaic, output_transform = merge.merge(ds_list)
    mosaic = np.zeros(mosaic.shape)
    out_meta = {
        'driver':'GTiff',
        'width': mosaic.shape[2],
        'height': mosaic.shape[1],
        'count': mosaic.shape[0],
        'dtype':'float64',
        'crs': crs,
        'transform': output_transform,
        'nodata':-9999.
    }

    out_path = os.path.join(out_root, 'temp_mosaic.tif')
    with rasterio.open(fp=out_path, mode='w', **out_meta) as dst:
        dst.write(mosaic)
    base_ds = rasterio.open(out_path)

    stack = np.zeros((mosaic.shape[1], mosaic.shape[2], len(ds_list)))
    for i, ds in enumerate(ds_list):
        print(i)
        mosaic, output_transform = merge.merge([ds, base_ds])
        stack[..., i] = mosaic

    os.remove(out_path)

    return stack, output_transform


def filter_stack_save(swot_wse_stack, swot_qual_stack, out_path, transform, crs, bad_data_value):
    bad_data = np.where(swot_qual_stack != 1)
    swot_wse_stack[bad_data[0], bad_data[1], bad_data[2]] = bad_data_value
    swot_wse_stack[swot_wse_stack == 0] = bad_data_value

    out_meta = {
        'driver':'GTiff',
        'width': swot_wse_stack.shape[1],
        'height': swot_wse_stack.shape[0],
        'count': swot_wse_stack.shape[2],
        'dtype':'float64',
        'crs': crs,
        'transform': transform,
        'nodata':-9999.
    }

    with rasterio.open(fp=out_path, mode='w', **out_meta) as dst:
        dst.write(np.moveaxis(swot_wse_stack, 2, 0))

    return rasterio.open(out_path)


def crop_and_save(ds, swot_box, crs, transform, out_path, descriptions):
    window = rasterio.windows.from_bounds(*swot_box.bounds, transform)
    transform = ds.window_transform(window)

    ds_window = ds.read(window=window)
    out_meta = {
        'driver':'GTiff',
        'width': ds_window.shape[2],
        'height': ds_window.shape[1],
        'count': ds_window.shape[0],
        'dtype':'float64',
        'crs': crs,
        'transform': transform,
        'nodata':-9999.
    }

    with rasterio.open(fp=out_path, mode='w', **out_meta) as dst:
        dst.descriptions = descriptions 
        dst.write(ds_window)

    return rasterio.open(out_path)


def process_data(emit_root, wl, swot_root, out_root, bad_data_value=-9999.):

    # Get overlapping areas in EMIT data
    emit_fps = glob.glob(os.path.join(emit_root, '*rfl.tif'))
    swot_wse_fps = glob.glob(os.path.join(swot_root, 'wse', '*.tif'))
    swot_qual_fps = glob.glob(os.path.join(swot_root, 'wse_qual', '*.tif'))

    emit_fp = emit_fps[0]
    emit_ds = rasterio.open(emit_fp)
    emit_crs = pyproj.CRS(emit_ds.meta['crs'])

    swot_wse_fp = swot_wse_fps[0]
    swot_wse_ds = rasterio.open(swot_wse_fp)
    swot_crs = pyproj.CRS(swot_wse_ds.meta['crs'])

    ####################################################################
    # EMIT DATA
    ####################################################################
    emit_fps_reproj = []
    for i, fp in enumerate(sorted(emit_fps)):
        pattern = '(EMIT_L2A_RFL)_\w*_(\d{8}T\d{6})'
        fp_list = fp.split('/')
        fp_name = fp_list.pop(-1)
        match = re.search(pattern, fp_name)
        if not match:
            raise ValueError('Could not parse fp for metadata')

        print(fp)
        print()
        emit_ds = upright_image(rasterio.open(fp), fp)

        out_name = fp_name.split('.')[0] + f'_epsg_{swot_crs.to_epsg()}.tif'
        out_path = os.path.join('/'.join(fp_list), out_name)
        emit_ds = reproject(emit_ds, emit_crs, swot_crs, out_path)

        box = geometry.box(*emit_ds.bounds)
        if not i:
            box_base = box 
        else:
            box_base = box_base.intersection(box)

        emit_fps_reproj.append(out_path)

    swot_box = box_base 

    # Write study area
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }
    shape_outroot = os.path.join(out_root, 'Area')
    os.makedirs(shape_outroot, exist_ok=True)
    with fiona.open(
        os.path.join(shape_outroot, 'emit_overlap_area.shp'), 'w', 
        'ESRI Shapefile', schema
    ) as c:
        ## If there are multiple geometries, put the "for" loop here
        c.write({
            'geometry': geometry.mapping(swot_box),
            'properties': {'id': 1},
        })

    emit_data = {
        'id': [],
        'year': [],
        'month': [],
        'day': [],
        'hour': [],
        'minute': [],
        'second': [],
        'crop_path': []
    }
    for fp in sorted(emit_fps_reproj):
        print(fp)
        emit_ds = rasterio.open(fp)

        fp_name = fp.split('/')[-1]
        out_name = fp_name.split('.')[0] + f'_epsg_{swot_crs.to_epsg()}_CROP.tif'
        out_emit = os.path.join(out_root, 'EMIT')
        os.makedirs(out_emit, exist_ok=True)
        out_path = os.path.join(out_emit, out_name)

        emit_ds = crop_and_save(
            emit_ds, 
            swot_box, 
            emit_ds.crs, 
            emit_ds.transform, 
            out_path,
            [str(w) for w in wl]
        )

        emit_data['id'].append(
            match.group(1) 
            + '_' + match.group(2)
        )
        dt = datetime.datetime.strptime(match.group(2), '%Y%m%dT%H%M%S')
        emit_data['year'].append(dt.year)
        emit_data['month'].append(dt.month)
        emit_data['day'].append(dt.day)
        emit_data['hour'].append(dt.hour)
        emit_data['minute'].append(dt.minute)
        emit_data['second'].append(dt.second)
        emit_data['crop_path'].append(out_path)

    emit_data = pandas.DataFrame(data=emit_data)

    ####################################################################
    # SWOT DATA
    ####################################################################
    # Rotate SWOT images
    swot_wse_ds_list = []
    swot_qual_ds_list = []
    swot_data = {
        'id': [],
        'year': [],
        'month': [],
        'day': [],
        'hour': [],
        'minute': [],
        'second': [],
    }
    for wse_fp, qual_fp in zip(sorted(swot_wse_fps), sorted(swot_qual_fps)):
        pattern = '(SWOT_L2_HR_Raster_100m_UTM.{3})_\w*_x_(\d{3}_\d{3}_\d{3}\w)_(\d{8}T\d{6})_(\d{8}T\d{6})'
        match = re.search(pattern, wse_fp.split('/')[-1])
        if not match:
            raise ValueError(f'Could not parse fp for metadata: {wse_fp}')

        print(wse_fp)
        print(qual_fp)
        print()
        swot_wse_ds_list.append(upright_image(rasterio.open(wse_fp), wse_fp))
        swot_qual_ds_list.append(upright_image(rasterio.open(qual_fp), qual_fp))

        swot_data['id'].append(
            match.group(1) 
            + match.group(2)
            + '_' + match.group(3) 
        )
        dt = datetime.datetime.strptime(match.group(3), '%Y%m%dT%H%M%S')
        swot_data['year'].append(dt.year)
        swot_data['month'].append(dt.month)
        swot_data['day'].append(dt.day)
        swot_data['hour'].append(dt.hour)
        swot_data['minute'].append(dt.minute)
        swot_data['second'].append(dt.second)

    swot_data = pandas.DataFrame(data=swot_data)

    # Stack all swot arrays into cube
    swot_wse_stack, stack_transform = stack_arrays(swot_wse_ds_list, swot_crs, out_root)
    swot_qual_stack, _ = stack_arrays(swot_qual_ds_list, swot_crs, out_root)

    # Filter out the bad data + save swot
    out_swot = os.path.join(out_root, 'SWOT')
    os.makedirs(out_swot, exist_ok=True)
    out_path = os.path.join(out_swot, f'SWOT_wse_stack_epsg_{swot_crs.to_epsg()}_CROP.tif')
    swot_stack = filter_stack_save(
        swot_wse_stack, 
        swot_qual_stack, 
        out_path, 
        stack_transform, 
        swot_crs, 
        bad_data_value
    )

    # Crop to EMIT overlap
    # Get date list
    dts = []
    for year, month, day in zip(
        swot_data['year'], swot_data['month'], swot_data['day']
    ):
        dts.append(
            str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
        )

    swot_stack_window = crop_and_save(
        swot_stack,
        swot_box,
        swot_crs,
        stack_transform,
        out_path,
        dts
    )

    swot_data['crop_path'] = out_path


@click.command(name='main')
@click.option('--swot_inroot')
@click.option('--swot_outroot')
@click.option('--emit_inroot')
@click.option('--emit_outroot')
@click.option('--crop_outroot')
@click.option('--bad_data_value', default=-9999.)
def main(swot_inroot, swot_outroot, emit_inroot, emit_outroot, crop_outroot, bad_data_value):
    # Convert SWOT data to geotiff
    print('Converting SWOT to Geotiff')
    convert_swot_to_geotiff(swot_inroot, swot_outroot, 'wse')
    convert_swot_to_geotiff(swot_inroot, swot_outroot, 'wse_qual')

    # Convert EMIT data to geotiff
    print('Converting EMIT to Geotiff')
    wl = convert_emit_to_geotiff(emit_inroot, emit_outroot)

    # Flip image if needed, reproject, stack and crop
    print('Processing data')
    process_data(emit_outroot, wl, swot_outroot, crop_outroot, bad_data_value)


if __name__ == '__main__':
    main()
