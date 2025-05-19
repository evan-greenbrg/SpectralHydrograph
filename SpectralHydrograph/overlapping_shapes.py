from datetime import datetime
import os
import re

import click
import geopandas
from matplotlib import pyplot as plt
import numpy as np
import pandas
from shapely import geometry


def find_utm_epsg(latitude, longitude):
    utm_zone = int((longitude + 180) / 6) + 1

    if latitude >= 0:
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone

    return epsg_code


def overlap(ogeo, geo, min_area):
    data = {'urls': []}
    geoms = []
    for i, geom_i in ogeo[-1].iterrows():
        print(f"{i} of {len(ogeo[-1])}")
        for j, geom_j in geo.iterrows():
            if geom_j['url'] in geom_i['urls']:
                continue

            olap = geom_i['geometry'].intersection(geom_j['geometry'])
            if olap.area:
                data['urls'].append(geom_i['urls'] + [geom_j['url']])
                geoms.append(olap)

    gpd = geopandas.GeoDataFrame(
        pandas.DataFrame(data=data),
        geometry=geoms
    )
    if len(gpd):
        longitude, latitude = gpd.iloc[0]['geometry'].centroid.xy
        epsg = find_utm_epsg(latitude[0], longitude[0])
        gpd['area'] = (
            gpd['geometry'].set_crs('epsg:4326').to_crs(f'epsg:{epsg}').area
        )
        gpd = gpd[gpd['area'] >= min_area]
        gpd['num_images'] = [len(row['urls']) for i, row in gpd.iterrows()]

    ogeo.append(gpd)

    return ogeo


def find_overlaps(geo, min_area=1e8):
    geo = geo.drop(columns=[
        'year', 
        'month',
        'day',
        'hour',
        'minute',
        'second',
    ])
    ogeo = geo.copy()
    ogeo['urls'] = [[i] for i in ogeo['url']]
    ogeo = ogeo.drop(columns=['url'])
    ogeo['area'] = ogeo['geometry'].area
    ogeo['num_images'] = 1
    ogeo = [ogeo]

    while len(ogeo[-1]):
        print(len(ogeo[-1]))
        ogeo = overlap(ogeo, geo, min_area)
    ogeo.pop(-1)

    return pandas.concat(ogeo).reset_index(drop=True)


def emit_swot_overlaps(emit_overlaps, swot):
    data = {
        'emit_idx': [],
        'emit_urls': [], 
        'swot_url': [], 
        'emit_img_num': []
    }
    geoms = []
    for i, emit_row in emit_overlaps.iterrows():
        print(f"{i} of {len(emit_overlaps)}")
        for j, swot_row in swot.iterrows():
            olap = emit_row['geometry'].intersection(swot_row['geometry'])
            if olap.area:
                data['emit_urls'].append(emit_row['urls'])
                data['emit_img_num'].append(emit_row['num_images'])
                data['swot_url'].append(swot_row['url'])
                data['emit_idx'].append(i)
                geoms.append(olap)

    gdf = geopandas.GeoDataFrame(
        pandas.DataFrame(data=data),
        geometry=geoms
    )

    # Aggregate
    data = {
        'emit_idx': [],
        'emit_urls': [],
        'emit_img_num': [],
        'swot_urls': [],
        'swot_geometry': [],
    }
    for idx, group in gdf.groupby('emit_idx'):
        print(f"{idx} of {len(gdf['emit_idx'].unique())}")
        swot_urls = []
        swot_geoms = []
        for i, row in group.iterrows():
            swot_urls.append(row['swot_url'])
            swot_geoms.append(row['geometry'])

        data['emit_idx'].append(group.iloc[0]['emit_idx'])
        data['emit_urls'].append(group.iloc[0]['emit_urls'])
        data['emit_img_num'].append(group.iloc[0]['emit_img_num'])
        data['swot_urls'].append(swot_urls)
        data['swot_geometry'].append(swot_geoms)

    gdf = geopandas.GeoDataFrame(
        pandas.DataFrame(data=data),
        geometry=emit_overlaps['geometry']
    )

    return gdf


@click.command("pull_overlaps")
@click.argument("emit_path")
@click.argument("swot_path")
@click.argument("out_root")
def main(emit_path, swot_path, out_root):

    emit = geopandas.read_file(emit_path)
    swot = geopandas.read_file(swot_path)

    drop_i = []
    for i, row in swot.iterrows():
        if '100m' not in row['url']:
            drop_i.append(i)
    swot = swot.drop(drop_i).reset_index(drop=True)

    emit_overlaps = find_overlaps(emit)
    emit_swot = emit_swot_overlaps(emit_overlaps, swot).set_crs('epsg:4326')
    for num in emit_swot['emit_img_num'].unique():
        emit_swot_out = emit_swot[
            emit_swot['emit_img_num'] == num
        ].reset_index(drop=True)
        out_name = f'emit_swot_overlap_{num}_images'
        out_path = os.path.join(out_root, out_name + '.gpkg')
        emit_swot_out.to_file(out_path, layer=out_name, driver="GPKG")


if __name__ == '__main__':
    main()
