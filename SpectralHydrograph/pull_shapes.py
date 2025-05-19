import os
from datetime import datetime
import click
import re
import earthaccess
from matplotlib import pyplot as plt
import numpy as np
import pandas
import geopandas
from shapely.geometry.polygon import orient
from shapely import geometry
import xarray as xr
import sys


auth = earthaccess.login(persist=True)
print(auth.authenticated)


def get_key(dic, key, res=None):
    if dic.get(key):
        return dic[key]

    else:
        for v in dic.values():
            if res:
                break

            elif isinstance(v, dict):
                res = get_key(v, key)

        return res


def parse_data(results):
    data = {
        'year': [],
        'month': [],
        'day': [],
        'hour': [],
        'minute': [],
        'second': [],
        'url': [],
    }
    polys = []
    for res in results:
        print(get_key(res['umm'], 'BeginningDateTime'))
        # Parse time
        try:
            dt = datetime.strptime(
                get_key(res['umm'], 'BeginningDateTime'), 
                '%Y-%m-%dT%H:%M:%SZ'
            )
        except:
            pattern = '(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})'
            dt_match = re.search(pattern, get_key(res['umm'], 'BeginningDateTime'))
            dt = datetime(
                int(dt_match.group(1)),
                int(dt_match.group(2)),
                int(dt_match.group(3)),
                int(dt_match.group(4)),
                int(dt_match.group(5)),
                int(dt_match.group(6)),
            )

        # Parse URLS
        url = get_key(res['umm'], 'GranuleUR')

        # Parse polygon
        coords = []
        polygon_points = get_key(res['umm'], 'GPolygons')
        if polygon_points:
            for points in polygon_points[0]['Boundary']['Points']:
                coords.append([points['Longitude'], points['Latitude']])
            coords = np.array(coords)
            poly = geometry.Polygon(coords)
            polys.append(poly)

            data['year'].append(dt.year)
            data['month'].append(dt.month)
            data['day'].append(dt.day)
            data['hour'].append(dt.hour)
            data['minute'].append(dt.minute)
            data['second'].append(dt.second)

            data['url'].append(url)

    df = pandas.DataFrame(data=data)
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=polys
    )

    return gdf


@click.command(name='pull_data')
@click.argument('poly_path')
@click.option('--start_date', help='2024-02-04', type=str)
@click.option('--end_date', help='2024-02-04', type=str)
@click.option('--max_cloud', type=int)
@click.option('--out_root', type=str)
def pull_data_shapefile(poly_path, start_date, end_date, max_cloud, out_root):

    # geopackage in EPSG:4326
    geojson = geopandas.read_file(poly_path)
    bbox = tuple(list(geojson.total_bounds))

    # Search example using bounding box
    emit_name = 'EMITL2ARFL'
    emit_results = earthaccess.search_data(
        short_name=emit_name,
        bounding_box=bbox,
        temporal=(start_date, end_date),
        cloud_cover=(0, max_cloud),
        count=-1
    )
    emit = parse_data(emit_results).set_crs('epsg:4326')
    out_path = os.path.join(out_root, 'emit_data.gpkg')
    emit.to_file(out_path, layer='emit_data', driver="GPKG")

    swot_name = 'SWOT_L2_HR_Raster_2.0'
    swot_results = earthaccess.search_data(
        short_name=swot_name,
        bounding_box=bbox,
        temporal=(start_date, end_date),
        count=-1
    )
    swot = parse_data(swot_results).set_crs('epsg:4326')
    out_path = os.path.join(out_root, 'swot_data.gpkg')
    swot.to_file(out_path, layer='swot_data', driver="GPKG")


if __name__ == '__main__':
    pull_data_shapefile()
