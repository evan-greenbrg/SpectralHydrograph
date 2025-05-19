import os
import click
import earthaccess
import geopandas


fs = earthaccess.get_requests_https_session()


def download(imid, dataset, out_root):
    if dataset == 'emit':
        url =  f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/{imid}/{imid}.nc'
    elif dataset == 'swot':
        url = f'https://archive.swot.podaac.earthdata.nasa.gov/podaac-swot-ops-cumulus-protected/SWOT_L2_HR_Raster_2.0/{imid}.nc'
    else:
        raise ValueError('No dataset supported/given')

    out_path = os.path.join(out_root, f'{imid}.nc')

    # Check if it's already been downloaded
    if os.path.exists(out_path):
        print('Already downloaded, skipping')
        return out_path

    # If not, download
    earthaccess.download(
        url, 
        local_path=out_root
    )

    return out_path


@click.command(name='download')
@click.argument('path')
@click.argument('out_root')
@click.option('--swot_code')
def download_all_images(path, out_root, swot_code):
    geo = geopandas.read_file(path)

    for i, row in geo.iterrows():
        emit_ids = eval(row['emit_urls'])
        swot_ids = eval(row['swot_urls'])

        out_path = os.path.join(out_root, 'EMIT', 'nc')
        os.makedirs(out_path)
        emit_paths = []
        for imid in emit_ids:
            print(f'Downloading: {imid}')
            emit_paths.append(download(imid, 'emit', out_path))

        out_path = os.path.join(out_root, 'SWOT', 'nc')
        os.makedirs(out_path)
        swot_paths = []
        for imid in swot_ids:
            print(imid)
            if swot_code in imid:
                print(f'Downloading: {imid}')
                swot_paths.append(download(imid, 'swot', out_path))


if __name__ == '__main__':
    download_all_images()
