import re
import glob
import os

import click
import geopandas
import pandas
from spectral import envi
from matplotlib import pyplot as plt
import numpy as np
import rasterio
from rasterio import merge
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from scipy import ndimage
from pyproj import Transformer
from shapely.ops import transform
from rasterio import features


def normalize(data, vmin=0, vmax=15):
    """
    Blue - Band 19
    Green - Band 30
    Red - Band 54
    NIR - Band 106
    SWIR - Band 165
    """
    norm_data = (data - vmin) / (vmax - vmin)

    return norm_data


def stack_images(emit_path, water_path):
    # Load ds
    emit_ds = rasterio.open(emit_path)
    water_ds = rasterio.open(water_path)

    # Merge ds
    with MemoryFile() as memfile:
        # Define the metadata for the single-band raster
        meta = emit_ds.meta.copy()
        meta.update({
            'count': 1,
            'driver': 'GTiff'
        })
        
        # Write the single band to the memory file
        with memfile.open(**meta) as mem:
            mem.write(emit_ds.read(1), 1)

            mosaic, output_transform = merge.merge([water_ds, mem])
            mosaic = np.zeros(mosaic.shape)
            out_meta = {
                'driver':'GTiff',
                'width': mosaic.shape[2],
                'height': mosaic.shape[1],
                'count': mosaic.shape[0],
                'dtype':'float64',
                'crs': emit_ds.crs,
                'transform': output_transform,
                'nodata':-9999.
            }

            base_path = os.path.join('/tmp/temp_mosaic.tif')
            with rasterio.open(fp=base_path , mode='w', **out_meta) as dst:
                dst.write(mosaic)

    base_ds = rasterio.open(base_path)

    emit_stack = []
    for i in range(emit_ds.count):
        with MemoryFile() as memfile:
            # Define the metadata for the single-band raster
            meta = emit_ds.meta.copy()
            meta.update({
                'count': 1,
                'driver': 'GTiff'
            })
            
            # Write the single band to the memory file
            with memfile.open(**meta) as mem:
                mem.write(emit_ds.read(i + 1), 1)
                mosaic, output_transform = merge.merge(
                    [mem, base_ds],
                    use_highest_res=True
                )

        emit_stack.append(mosaic)

    emit_stack = np.squeeze(np.array(emit_stack))
    emit_transform = output_transform

    # Get baseline
    water_stack, water_transform = merge.merge(
        [water_ds, base_ds],
        use_highest_res=True
    )

    return (emit_stack, emit_transform), (water_stack, water_transform)


def pull_spectra(emit_data, water_data, roi_raster, out_root, out_name, wl, plot=False):
    # Reduce chance of false positives
    water = ndimage.binary_erosion(np.squeeze(water_data))
    row, col = np.where(roi_raster == 0)
    water[row, col] = 0

    row, col = np.where(water)
    sample = emit_data[:, row, col].T

    good_idx = []
    for i, s in enumerate(sample):
        if np.any(s != 0):
            good_idx.append(i)
    sample = sample[good_idx]

    # Get some stats
    mean = np.nanmean(sample, axis=0)
    std = np.nanstd(sample, axis=0)
    median = np.nanmedian(sample, axis=0)
    maxs = np.nanmax(sample, axis=0)
    mins = np.nanmin(sample, axis=0)

    if plot:
        fig, axs = plt.subplots(2, 1)
        bands = [50, 30, 18]
        plot_im = np.moveaxis(emit_data[bands, ...], 0, -1)
        plot_im[row, col, :] = 1
        axs[0].imshow(normalize(
            plot_im,
            vmin=0, vmax=.3
        ))

        axs[1].fill_between(wl, mins, maxs, color='lightgray', alpha=.5)
        axs[1].fill_between(
            wl, 
            median - (2 * std), 
            median + (2 * std), 
            color='lightblue', alpha=.5
        )
        axs[1].plot(wl, median, color='black')
        axs[1].set_xlabel('Wavelength')
        axs[1].set_ylabel('Reflectance')
        plt.show()

    # Construct outputs
    df = pandas.DataFrame(sample).transpose()
    df.index = wl
    df = df.reset_index(drop=False, names=['wl'])

    out_path = os.path.join(out_root, out_name + '_sample.csv')
    df.to_csv(out_path)

    df = pandas.DataFrame(data={
        'wl': wl,
        'mean': mean,
        'std': std,
        'median': maxs,
        'mins': mins,
    })
    out_path = os.path.join(out_root, out_name + '_stats.csv')
    df.to_csv(out_path)


@click.command(name='main')
@click.argument('emit_path')
@click.argument('water_path')
@click.argument('roi_path')
@click.argument('out_root')
def main(emit_path, water_path, roi_path, out_root):
    match = re.search('(EMIT_L2A_RFL_.{3}_\d{8}T\d{6})', emit_path)
    if match:
        out_name = match.groups(1)[0]
    else:
        raise ValueError('Could not match file id. Check.')

    emit_ds = rasterio.open(emit_path)
    wl = np.array(emit_ds.descriptions).astype(float)

    (emit_data, emit_transform), (water_data, water_transform) = stack_images(
        emit_path,
        water_path
    )

    # Reproject ROI to swot epsg
    roi = geopandas.read_file(roi_path).iloc[0]['geometry']
    transformer = Transformer.from_crs(
        "EPSG:4326", emit_ds.crs, always_xy=True
    )
    roi = transform(transformer.transform, roi)

    roi_raster = features.rasterize(
        [(roi, 1)], 
        out_shape=(water_data.shape[1], water_data.shape[2]),
        transform=water_transform,
        fill=0,
        dtype='uint8'
    )

    pull_spectra(
        emit_data, 
        water_data,
        roi_raster,
        out_root, 
        out_name, 
        wl, 
        plot=True
    )


if __name__ == '__main__':
    main()
