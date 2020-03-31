#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os.path
from osgeo import gdal
import numpy as np


#bioclim_variables = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
#                     '17', '18', '19']

baltic_years = list(range(1991,2018))

#perhaps from .txt file?
baltic_vars = ['Clupea_harengus_JUV','Clupea_harengus_SA','Clupea_harengus_LA','Gadus_morhua_JUV','Gadus_morhua_SA',
               'Gadus_morhua_LA', 'Sprattus_sprattus_JUV','Sprattus_sprattus_SA','Sprattus_sprattus_LA', 'TEMP', 'SALIN',
               'TOTOXY','TOTP','TOTN','SIO4','NO3N','NH4N','COPEPOD']

baltic_depth = ['(0, 20)','(20, 70)','(70, 460)']



def get_array_from_tif(path, year, var, depth):
    if depth != None
        ds = gdal.Open(os.path.join(path, '{}_{}_{}_cut.tif'.format(year,var,depth)))
        band = ds.GetRasterBand(1)
    else:
        ds = gdal.Open(os.path.join(path, '{}_{}_cut.tif'.format(year,var)))
    return band.ReadAsArray()


def get_value_from_array(lat, lon, cci_array, array_height, array_width):
    lat_pos = int((array_height / 2) - (lat * (array_height / 180)))
    lon_pos = int((array_width / 2) + (lon * (array_width / 360)))
    value = cci_array[lat_pos][lon_pos]
    if value < -1000000:
        return np.nan
    return value


### What to do with this function as for baltic case there are so many layer names? But all in separate raster maps,
### Can this be done from txt file listing all layers instead?
def get_layer_names(_):
    return ['BioClim_01', 'BioClim_02', 'BioClim_03', 'BioClim_04', 'BioClim_05', 'BioClim_06', 'BioClim_07',
            'BioClim_08', 'BioClim_09', 'BioClim_10', 'BioClim_11', 'BioClim_12', 'BioClim_13', 'BioClim_14',
            'BioClim_15', 'BioClim_16', 'BioClim_17', 'BioClim_18', 'BioClim_19']


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
    array = get_array_from_tif(path, layer_name[-2:])
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

