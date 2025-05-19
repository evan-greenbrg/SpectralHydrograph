import os
import datetime
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio import merge
import click
from rasterio import features
from rasterio.io import MemoryFile
import pandas
import geopandas
from pyproj import Transformer
from shapely.ops import transform
from shapely import geometry
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt


resampling_strategy = {
    'nearest': Resampling.nearest,
    'bilinear': Resampling.bilinear,
    'cubic': Resampling.cubic,
    'lanczos': Resampling.lanczos,
}


def resample_to_match(
    water_path, 
    swot_path, 
    out_path, 
    resampling_method='bilinear'
):
    input_ds = rasterio.open(water_path)

    input_ds = rasterio.open(water_path)
    ref_ds = rasterio.open(swot_path)

    # Define the new resolution (e.g., if n = 50, set it to (50, 50))
    new_res = ref_ds.res

    # Calculate the scaling factors
    scale_x = input_ds.res[0] / new_res[0]
    scale_y = input_ds.res[1] / new_res[1]

    # Compute the new dimensions
    new_width = int(input_ds.width * scale_x)
    new_height = int(input_ds.height * scale_y)

    # Update the transform for the new resolution
    new_transform = input_ds.transform * input_ds.transform.scale(
        1 / scale_x,
        1 / scale_y
    )

    # Perform the resampling
    data = input_ds.read(
        out_shape=(input_ds.count, new_height, new_width),
        resampling=resampling_strategy.get(
            resampling_method, Resampling.bilinear
        )
    )

    # Update metadata for the new raster
    new_meta = input_ds.meta.copy()
    new_meta.update({
        'transform': new_transform,
        'width': new_width,
        'height': new_height
    })

    # Write the resampled raster to disk
    with rasterio.open(out_path, 'w', **new_meta) as dest:
        dest.write(data)

    return out_path


def stack_water_wse(swot_ds, water_ds):

    # Create basesize
    with MemoryFile() as memfile:
        # Define the metadata for the single-band raster
        meta = swot_ds.meta.copy()
        meta.update({
            'count': 1,
            'driver': 'GTiff'
        })
        
        # Write the single band to the memory file
        with memfile.open(**meta) as mem:
            mem.write(swot_ds.read(1), 1)

            mosaic, output_transform = merge.merge([water_ds, mem])
            mosaic = np.zeros(mosaic.shape)
            out_meta = {
                'driver':'GTiff',
                'width': mosaic.shape[2],
                'height': mosaic.shape[1],
                'count': mosaic.shape[0],
                'dtype':'float64',
                'crs': swot_ds.crs,
                'transform': output_transform,
                'nodata':-9999.
            }

            base_path = os.path.join('/tmp/temp_mosaic.tif')
            with rasterio.open(fp=base_path , mode='w', **out_meta) as dst:
                dst.write(mosaic)

    base_ds = rasterio.open(base_path)

    # Create the SWOT Stack
    swot_stack = []
    for i in range(swot_ds.count):
        with MemoryFile() as memfile:
            # Define the metadata for the single-band raster
            meta = swot_ds.meta.copy()
            meta.update({
                'count': 1,
                'driver': 'GTiff'
            })
            
            # Write the single band to the memory file
            with memfile.open(**meta) as mem:
                mem.write(swot_ds.read(i + 1), 1)
                mosaic, output_transform = merge.merge([mem, base_ds])

        swot_stack.append(mosaic)

    swot_stack = np.squeeze(np.array(swot_stack))
    swot_transform = output_transform

    # Create WATER Stack
    water_stack, water_transform = merge.merge([water_ds, base_ds])

    os.remove(base_path)

    return (swot_stack, swot_transform), (water_stack, water_transform) 


def sample_swot(swot_path, water_path, roi_path, out_path):
    swot_ds = rasterio.open(swot_path)
    water_ds = rasterio.open(water_path)

    # Stack to same grid
    (
        (swot_stack, swot_transform), 
        (water_stack, water_transform) 
    ) = stack_water_wse(swot_ds, water_ds)

    roi = geopandas.read_file(roi_path).iloc[0]['geometry']

    # Reproject ROI to swot epsg
    transformer = Transformer.from_crs("EPSG:4326", swot_ds.crs, always_xy=True)
    roi = transform(transformer.transform, roi)

    # Get the sample idx
    raster = features.rasterize(
        [(roi, 1)], 
        out_shape=(water_stack.shape[1], water_stack.shape[2]),
        transform=water_transform,
        fill=0,
        dtype='uint8'
    )
    row, col = np.where(raster)

    data = {
        'year': [],
        'mean': [],
        'median': [],
        'std': [],
        'min': [],
        'max': [],
    }
    years = swot_ds.descriptions
    for i in range(swot_ds.count):
        sample = swot_stack[i, row, col]
        sample = sample[sample > 0]
        if len(sample):
            data['year'].append(years[i])
            data['mean'].append(np.nanmean(sample))
            data['median'].append(np.nanmedian(sample))
            data['std'].append(np.nanstd(sample))
            data['min'].append(np.nanmin(sample))
            data['max'].append(np.nanmax(sample))

    df = pandas.DataFrame(data=data)
    df['dt'] = pandas.to_datetime(df['year'])
    df.to_csv(out_path)

    plt.fill_between(
        df['dt'],
        df['median'] - (2 * df['std']),
        df['median'] + (2 * df['std']),
        color='blue',
        alpha=.5
    )
    plt.scatter(
        df['dt'], 
        df['median'],
        s=60,
        edgecolor='black', facecolor='white'
    )
    plt.show()


@click.command(name='main')
@click.argument('swot_path')
@click.argument('water_path')
@click.argument('roi_path')
@click.option('--swot_water_out')
@click.option('--data_out')
def main(swot_path, water_path, roi_path, swot_water_out, data_out):
    #Resample the water classification to SWOT
    print('Resampling water classification')
    swot_water_out_path = resample_to_match(
        water_path,
        swot_path,
        swot_water_out
    )

    print('Sampling SWOT data')
    sample_swot(
        swot_path, 
        swot_water_out_path, 
        roi_path, 
        data_out
    )


if __name__ == '__main__':
    main()

