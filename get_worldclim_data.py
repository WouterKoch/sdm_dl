import os.path
import sys
from osgeo import gdal
import numpy as np

month_codes = [None, '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
worldclim_variables = ['prec', 'srad', 'tavg', 'tmax', 'tmin', 'vapr', 'wind']


def get_layer_names(_):
    return [
                    'WorldClim_vapr_m01', 'WorldClim_vapr_m02', 'WorldClim_vapr_m03', 'WorldClim_vapr_m04',
                    'WorldClim_vapr_m05', 'WorldClim_vapr_m06', 'WorldClim_vapr_m07', 'WorldClim_vapr_m08',
                    'WorldClim_vapr_m09', 'WorldClim_vapr_m10', 'WorldClim_vapr_m11', 'WorldClim_vapr_m12',
            ]


def get_array_from_tif(path_to_tif):
    ds = gdal.Open(path_to_tif)
    band = ds.GetRasterBand(1)
    return band.ReadAsArray()


def get_value_from_array(lat, lon, tif_array, array_height, array_width):
    lat_pos = int((array_height / 2) - (lat * (array_height / 180)))
    lon_pos = int((array_width / 2) + (lon * (array_width / 360)))
    value = tif_array[lat_pos][lon_pos]
    if value < -10000 or value == 65535:
        return np.nan
    return value


def fill_blocks(layer_name, to_fetch, cell_size_degrees):
    for block in to_fetch:
        if block != None:
            break
        return [None] * len(to_fetch)

    month = layer_name[-2:]
    worldclim_variable = layer_name[(-(len(layer_name) - 10)):][:-4]

    path = os.getenv("WORLDCLIM_FOLDER_PATH")
    array = get_array_from_tif(
        os.path.join(path, worldclim_variable, 'wc2.0_30s_' + worldclim_variable + '_' + month + '.tif'))
    array_height, array_width = array.shape
    result = []

    for block_index in range(len(to_fetch)):
        if to_fetch[block_index] == None:
            result += [None]
            continue
        block = to_fetch[block_index]['block']
        block_height, block_width = block.shape
        lat_start, lon_start = to_fetch[block_index]['lat_lon_start']

        for row in range(block_height):
            for col in range(block_width):
                if np.isnan(block[row, col]):
                    block[row, col] = get_value_from_array(lat_start - (row * cell_size_degrees),
                                                           lon_start + (col * cell_size_degrees), array,
                                                           array_height,
                                                           array_width)
        result += [block]

    return result