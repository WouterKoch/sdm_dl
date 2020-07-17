import os.path

import datetime
import numpy as np
from osgeo import gdal
from typing import List

from layer_readers.general import AbstractLayerReader, get_raster


class LayerReader(AbstractLayerReader):
    worldclim_variables = ['prec', 'srad', 'tavg', 'tmax', 'tmin', 'vapr', 'wind']

    def get_layer_names(self, date: datetime.datetime) -> List[str]:
        return [f"WorldClim_{variable_name}_m{date.month:02}" for variable_name in self.worldclim_variables]

    def get_layer_from_file(self, layer_name):
        filename = os.path.join(os.getenv("RASTER_CACHE_FOLDER_PATH"), 'worldclim', layer_name + '.npz')

        raster_cell_size_deg, raster_max_lat, raster_max_lon, raster_min_lat, raster_min_lon = get_raster()

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


    def get_array_from_tif(self, path_to_tif):
        ds = gdal.Open(path_to_tif)
        band = ds.GetRasterBand(1)
        return band.ReadAsArray()

    def get_value_from_array(self, lat, lon, tif_array, array_height, array_width):
        lat_pos = int((array_height / 2) - (lat * (array_height / 180)))
        lon_pos = int((array_width / 2) + (lon * (array_width / 360)))
        value = tif_array[lat_pos][lon_pos]
        if value < -10000 or value == 65535:
            return np.nan
        return value

    def fill_blocks(self, layer_name, to_fetch, cell_size_degrees):
        """

        :param layer_name: one of the layer names returned by `get_layer_names`, e.g. 'WorldClim_vapr_m02'
        :param to_fetch:
        :param cell_size_degrees:
        :return:
        """
        for block in to_fetch:
            if block != None:
                break
            return [None] * len(to_fetch)

        import re
        match = re.match(r".*_(.*)_m(\d+)", layer_name)
        if match is None:
            raise ValueError(f"Incompatible layer name '{layer_name}'")
        worldclim_variable, month = match.groups()

        path = os.getenv("WORLDCLIM_FOLDER_PATH")
        resolution = "10m"
        version = "2.1"
        filename = f"wc{version}_{resolution}_" + worldclim_variable + "_" + month + ".tif"
        full_path = os.path.join(path, worldclim_variable, filename)
        print(full_path)
        array = self.get_array_from_tif(full_path)
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
                        block[row, col] = self.get_value_from_array(lat_start - (row * cell_size_degrees),
                                                                    lon_start + (col * cell_size_degrees), array,
                                                                    array_height,
                                                                    array_width)
            result += [block]

        return result
