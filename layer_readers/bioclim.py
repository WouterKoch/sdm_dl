import os.path

import numpy as np
from osgeo import gdal

from layer_readers.general import AbstractLayerReader, get_raster


class LayerReader(AbstractLayerReader):
    bioclim_variables = [str(x) for x in range(1,
                                               20)]  # ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']

    def get_array_from_tif(self, path, bioclim_variable):
        layer_path = os.path.join(path, 'wc2.1_10m_bio_' + str(int(bioclim_variable)) + '.tif')
        print(layer_path, os.path.exists(layer_path))
        ds = gdal.Open(layer_path)
        band = ds.GetRasterBand(1)
        return band.ReadAsArray()

    def get_value_from_array(self, lat, lon, cci_array, array_height, array_width):
        lat_pos = int((array_height / 2) - (lat * (array_height / 180)))
        lon_pos = int((array_width / 2) + (lon * (array_width / 360)))
        value = cci_array[lat_pos][lon_pos]
        if value < -1000000:
            return np.nan
        return value

    def get_layer_names(self, _):
        return ['BioClim_01', 'BioClim_02', 'BioClim_03', 'BioClim_04', 'BioClim_05', 'BioClim_06', 'BioClim_07',
                'BioClim_08', 'BioClim_09', 'BioClim_10', 'BioClim_11', 'BioClim_12', 'BioClim_13', 'BioClim_14',
                'BioClim_15', 'BioClim_16', 'BioClim_17', 'BioClim_18', 'BioClim_19']

    def get_layer_from_file(self, layer_name):
        filename = os.path.join(os.getenv("RASTER_CACHE_FOLDER_PATH"), 'bioclim', layer_name + '.npz')

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

    def fill_blocks(self, layer_name, to_fetch, cell_size_degrees):
        for block in to_fetch:
            if block is not None:
                break
            return [None] * len(to_fetch)

        path = os.getenv("BIOCLIM_FOLDER_PATH")
        array = self.get_array_from_tif(path, layer_name[-2:])
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
