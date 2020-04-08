#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os.path
from osgeo import gdal
import numpy as np

baltic_years = list(range(1991,2018))

#perhaps from .txt file? / get_layer_names kind of does the same?
baltic_vars = ['Clupea_harengus_JUV','Clupea_harengus_SA','Clupea_harengus_LA','Gadus_morhua_JUV','Gadus_morhua_SA',
               'Gadus_morhua_LA', 'Sprattus_sprattus_JUV','Sprattus_sprattus_SA','Sprattus_sprattus_LA', 'TEMP', 'SALIN',
               'TOTOXY','TOTP','TOTN','SIO4','NO3N','NH4N','COPEPOD']

baltic_depth = ['(0, 20)','(20, 70)','(70, 460)']

def get_array_from_tif(_, baltic_variable):
    path = os.environ["BALTIC_FILE_PATH"]
    a = os.path.join(path, baltic_variable + '.tif')
    ds = gdal.Open(os.path.join(path, baltic_variable + '.tif'))
    band = ds.GetRasterBand(1)
    b_array = band.ReadAsArray()
    b_stats = [np.min(b_array),np.max(b_array),"shape: {}".format(np.shape(b_array))]
    print(a)
    print(b_stats)
    return band.ReadAsArray()

def get_value_from_array(lat, lon, cci_array, array_height, array_width):
    lat_pos = int((array_height / 2) - (lat * (array_height / 180)))
    lon_pos = int((array_width / 2) + (lon * (array_width / 360)))
    value = cci_array[lat_pos][lon_pos]
    if value < -1000000:
        return np.nan
    return value

def get_layer_names(time):
    filenames = os.listdir(os.environ["BALTIC_FILE_PATH"])
    suffix = '.tif'
    year = str(time.year)
    layer_names = []

    for fn in filenames:
        fn = fn.replace(suffix, '')
        layer_names.append(fn)
    return [layer_name for layer_name in layer_names if year in layer_name]

def get_layer_from_file(layer_name):
    filename = os.path.join(os.getenv("RASTER_CACHE_FOLDER_PATH"), 'baltic', layer_name + '.npz')

    raster_max_lat = int(os.getenv("RASTER_MAX_LAT"))
    raster_min_lat = int(os.getenv("RASTER_MIN_LAT"))
    raster_max_lon = int(os.getenv("RASTER_MAX_LON"))
    raster_min_lon = int(os.getenv("RASTER_MIN_LON"))
    raster_cell_size_deg = float(os.getenv("RASTER_CELL_SIZE_DEG"))

    layer = {}

    if os.path.exists(filename):
        layer = dict(np.load(filename, allow_pickle=True))
        layer['metadata'] = layer['metadata'].item()
        print(layer['metadata'])
    else:
        raster_width = int((raster_max_lon - raster_min_lon) / raster_cell_size_deg) + 1
        raster_height = int((raster_max_lat - raster_min_lat) / raster_cell_size_deg) + 1
        layer['metadata'] = {'lat_NW_cell_center': raster_max_lat, 'lon_NW_cell_center': raster_min_lon,
                             'cell_size_degrees': raster_cell_size_deg, 'data_type': 'numerical',
                             'normalization_range': (99999999998, -99999999998), 'null_value': -99999999999}
        layer['map'] = np.ones((raster_width, raster_height)) * np.nan

    layer['filename'] = filename
    return layer


def fill_blocks(layer_name, to_fetch, cell_size_degrees):
    for block in to_fetch:
        if block is not None:
            break
        return [None] * len(to_fetch)

    path = os.getenv("BALTIC_FOLDER_PATH")
    print(layer_name)
    array = get_array_from_tif(path, layer_name)#[-2:])
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


