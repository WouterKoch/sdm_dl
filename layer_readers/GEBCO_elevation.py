import os
from builtins import print

from tqdm import tqdm
import numpy as np
from struct import unpack
from netCDF4 import Dataset


class LayerReader:
    current_file = None
    current_file_character = None

    def get_value(self, lat, lon, current_file):
        if current_file == None:
            file = Dataset(os.path.join(os.getenv("GEBCO_FOLDER_PATH"), 'GEBCO_2020.nc'), 'r')
            current_file = file.variables['elevation']

        return current_file[int((90 + lat) * 60 * 4)][int((180 + lon) * 60 * 4)], current_file

    def get_layer_names(self, _):
        return ['elevation']

    def get_layer_from_file(self, layer_name):
        filename = os.path.join(os.getenv("RASTER_CACHE_FOLDER_PATH"), 'GEBCO', layer_name + '.npz')

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
            layer['map'] = np.ones((raster_height, raster_width)) * np.nan

        layer['filename'] = filename
        return layer

    def fill_blocks(self, _, to_fetch, cell_size_degrees):
        fetched = []

        for block_index in range(len(to_fetch)):
            if to_fetch[block_index] == None:
                fetched += [None]
                continue
            block = to_fetch[block_index]['block']
            block_height, block_width = block.shape
            lat_start, lon_start = to_fetch[block_index]['lat_lon_start']

            current_file = None
            with tqdm(total=block_height * block_width, position=0, leave=True) as pbar:
                for row in range(block_height):
                    for col in range(block_width):
                        if np.isnan(block[row, col]):
                            block[row, col], current_file = self.get_value(
                                lat_start - (row * cell_size_degrees),
                                lon_start + (col * cell_size_degrees),
                                current_file)
                        pbar.update(1)
            fetched += [block]
            pbar.close()
        return fetched
