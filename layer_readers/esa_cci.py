import os.path
from osgeo import gdal
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np

class LayerReader:
    cci_codes = [10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121, 122, 130, 140,
                 150,
                 151, 152, 153, 160, 170, 180, 190, 200, 201, 202, 210, 220]

    def get_layer_names(self, date):
        year = str(min(max(date.year, 1992), 2018))
        return ['ESA_CCI_' + year]


    def get_layer_from_file(self, layer_name):
        filename = os.path.join(os.getenv("RASTER_CACHE_FOLDER_PATH"), 'esacci', layer_name + '.npz')

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
                                 'cell_size_degrees': raster_cell_size_deg, 'data_type': 'categorical',
                                 'data_categories': [10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90,
                                                     100, 110, 120,
                                                     121, 122, 130, 140, 150, 151, 152, 153, 160, 170, 180, 190, 200, 201,
                                                     202, 210,
                                                     220],
                                 'data_labels': {
                                     0: 'No data',
                                     10: 'Cropland, rainfed',
                                     11: 'Herbaceous cover',
                                     12: 'Tree or shrub cover',
                                     20: 'Cropland, irrigated or post-flooding',
                                     30: 'Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)',
                                     40: 'Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%) ',
                                     50: 'Tree cover, broadleaved, evergreen, closed to open (>15%)',
                                     60: 'Tree cover, broadleaved, deciduous, closed to open (>15%)',
                                     61: 'Tree cover, broadleaved, deciduous, closed (>40%)',
                                     62: 'Tree cover, broadleaved, deciduous, open (15-40%)',
                                     70: 'Tree cover, needleleaved, evergreen, closed to open (>15%)',
                                     71: 'Tree cover, needleleaved, evergreen, closed (>40%)',
                                     72: 'Tree cover, needleleaved, evergreen, open (15-40%)',
                                     80: 'Tree cover, needleleaved, deciduous, closed to open (>15%)',
                                     81: 'Tree cover, needleleaved, deciduous, closed (>40%)',
                                     82: 'Tree cover, needleleaved, deciduous, open (15-40%)',
                                     90: 'Tree cover, mixed leaf type (broadleaved and needleleaved)',
                                     100: 'Mosaic tree and shrub (>50%) / herbaceous cover (<50%)',
                                     110: 'Mosaic herbaceous cover (>50%) / tree and shrub (<50%)',
                                     120: 'Shrubland',
                                     121: 'Shrubland evergreen',
                                     122: 'Shrubland deciduous',
                                     130: 'Grassland',
                                     140: 'Lichens and mosses',
                                     150: 'Sparse vegetation (tree, shrub, herbaceous cover) (<15%)',
                                     151: 'Sparse tree (<15%)',
                                     152: 'Sparse shrub (<15%)',
                                     153: 'Sparse herbaceous cover (<15%)',
                                     160: 'Tree cover, flooded, fresh or brakish water',
                                     170: 'Tree cover, flooded, saline water',
                                     180: 'Shrub or herbaceous cover, flooded, fresh/saline/brakish water',
                                     190: 'Urban areas',
                                     200: 'Bare areas',
                                     201: 'Consolidated bare areas',
                                     202: 'Unconsolidated bare areas',
                                     210: 'Water bodies',
                                     220: 'Permanent snow and ice'
                                 }
                , 'null_value': -99999999999}
            layer['map'] = np.ones((raster_height, raster_width)) * np.nan

        layer['filename'] = filename
        return layer


    def get_array(self, path, year):
        if year < 2016:
            tif_file = gdal.Open(os.path.join(path, 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-' + str(year) + '-v2.0.7.tif'))
            array = tif_file.GetRasterBand(1).ReadAsArray()
            array_height, array_width = array.shape
        else:
            file = Dataset(os.path.join(path, 'C3S-LC-L4-LCCS-Map-300m-P1Y-' + str(year) + '-v2.1.1.nc'), 'r')
            array = file.variables['lccs_class']
            array_height = file.dimensions['lat'].size
            array_width = file.dimensions['lon'].size
        return array, array_width, array_height


    def get_value(self, lat, lon, layer, array, array_height, array_width, path):
        year = int(layer[-4:])

        if array is None:
            array, array_width, array_height = self.get_array(path, year)

        lat_pos = int((array_height / 2) - (lat * (array_height / 180)))
        lon_pos = int((array_width / 2) + (lon * (array_width / 360)))

        if year < 2016:
            return int(array[lat_pos][lon_pos]), array, array_width, array_height
        else:
            return int(array[0, lat_pos, lon_pos]), array, array_width, array_height


    def fill_blocks(self, layer, to_fetch, cell_size_degrees):
        result = []

        path = os.getenv("ESACCI_FOLDER_PATH")

        for block_index, fetching in enumerate(to_fetch):
            if fetching == None:
                result += [None]
                continue
            GLOBE_elevation.py
            block = to_fetch[block_index]['block']
            block_height, block_width = block.shape
            lat_start, lon_start = to_fetch[block_index]['lat_lon_start']

            current_file, array_height, array_width = None, 0, 0
            with tqdm(total=block_height * block_width, position=0, leave=True) as pbar:
                for row in range(block_height):
                    for col in range(block_width):
                        if np.isnan(block[row, col]):
                            block[row, col], current_file, array_height, array_width = self.get_value(
                                lat_start - (row * cell_size_degrees),
                                lon_start + (col * cell_size_degrees),
                                layer, current_file, array_height, array_width, path)
                        pbar.update(1)
            result += [block]
            pbar.close()
        return result
