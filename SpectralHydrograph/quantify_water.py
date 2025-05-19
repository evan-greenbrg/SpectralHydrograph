import os
import click
import rasterio
import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from scipy import ndimage


def find_largest_sharding(lenx, max_devices):
    devices = max_devices
    while lenx % devices:
        devices -= 1

    return devices


def filter_image(out, thresh=100):
    """
    Temporary function to clean the image.
    Memory intensive.
    """
    masks = []
    for i in np.unique(out):
        temp = out.copy()
        temp[temp == i] = 9999
        temp[temp < 9999] = 0
        temp[temp == 9999] = 1
        label, n = ndimage.label(temp)
        sizes = ndimage.sum(temp, label, range(n + 1))
        mask = sizes >= thresh
        masks.append(mask[label])

    for i, mask in enumerate(masks):
        if not i:
            final = mask.astype(int) * (i + 1)
        else:
            final += mask.astype(int) * (i + 1)

    label, n = ndimage.label(final == 0)
    for i in np.unique(label):
        if not i:
            continue

        temp = label.copy()
        temp[temp != i] = 0
        temp[temp > 0] = 1

        vals, counts = np.unique(
            final[np.where(ndimage.binary_dilation(temp).astype(int) - temp)],
            return_counts=True,
        )
        final[label == i] = vals[np.argmax(counts)]

    return final - 1


@jax.jit
def calculate_ndwi(spec):
    return (spec[21] - spec[59]) / (spec[21] + spec[59])


@click.command(name='classify')
@click.argument('path')
@click.argument('out_path')
@click.option('--thresh', default=0.2)
@click.option('--ncpu', default=15)
@click.option('--filt_size', default=1000)
def main(path, out_path, thresh, ncpu, filt_size):
    jax.config.update('jax_num_cpu_devices', ncpu)
    ds = rasterio.open(path)
    wl = np.array(ds.descriptions).astype(float)
    im = np.moveaxis(ds.read(), 0, -1)

    print('Classifing')
    tree = jnp.array(
        np.reshape(im, (im.shape[0] * im.shape[1], im.shape[2]))
    ).T

    P = PartitionSpec
    mesh = jax.make_mesh((1, find_largest_sharding(tree.shape[1], ncpu)), ('a', 'b'))
    y = jax.device_put(tree, NamedSharding(mesh, P('a', 'b')))
    # jax.debug.visualize_array_sharding(y)

    res = jax.tree_util.tree_map(calculate_ndwi, y)
    kim = jnp.reshape(jnp.array(res), (im.shape[0], im.shape[1], 1))

    # thresh = 0.2
    water = (kim > thresh).astype(int)

    print('Filtering')

    water_filt = filter_image(np.array(water), thresh=filt_size)

    out_meta = {
        'driver':'GTiff',
        'height': water_filt.shape[0],
        'width': water_filt.shape[1],
        'count': water_filt.shape[2],
        'dtype':'int16',
        'crs': ds.crs,
        'transform': ds.transform,
        'nodata':-9999.
    }
    with rasterio.open(fp=out_path, mode='w', **out_meta) as dst:
        dst.write(np.moveaxis(water_filt, 2, 0))


if __name__ == '__main__':
    main()
