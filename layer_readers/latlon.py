import os
import numpy as np


class LayerReader:

    def get_layer_from_file(self, layer_name):
        raster_max_lat = int(os.getenv("RASTER_MAX_LAT"))
        raster_min_lat = int(os.getenv("RASTER_MIN_LAT"))
        raster_max_lon = int(os.getenv("RASTER_MAX_LON"))
        raster_min_lon = int(os.getenv("RASTER_MIN_LON"))
        raster_cell_size_deg = float(os.getenv("RASTER_CELL_SIZE_DEG"))

        raster_width = int((raster_max_lon - raster_min_lon) / raster_cell_size_deg) + 1
        raster_height = int((raster_max_lat - raster_min_lat) / raster_cell_size_deg) + 1

        # Switch dimensions so we can transpose later
        if layer_name == 'lat':
            array_width = raster_height
            array_height = raster_width
            array_min = raster_min_lat
            array_max = raster_max_lat
        else:
            array_width = raster_width
            array_height = raster_height
            array_min = raster_min_lon
            array_max = raster_max_lon

        array = np.tile(np.linspace(array_min, array_max, num=array_width), array_height).reshape(array_height, array_width)

        if layer_name == 'lat':
            array = np.fliplr(array)
            array = array.transpose()

        return {'metadata': {'lat_NW_cell_center': raster_max_lat, 'lon_NW_cell_center': raster_min_lon,
                             'cell_size_degrees': raster_cell_size_deg, 'data_type': 'numerical',
                             'normalization_range': (array_max, -array_min), 'null_value': -99999999999},
                'map': array,
                'filename': os.path.join(os.getenv("RASTER_CACHE_FOLDER_PATH"), 'latlon', layer_name + '.npz')
                }


    def get_layer_names(self, _):
        return ['lat', 'lon']


    def fill_blocks(self, _, blocks, ___):
        return blocks
