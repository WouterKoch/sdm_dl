import os.path
from osgeo import gdal
from tqdm import tqdm
from netCDF4 import Dataset
import numpy as np

cci_codes = [10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121, 122, 130, 140, 150,
             151, 152, 153, 160, 170, 180, 190, 200, 201, 202, 210, 220]


def get_layer_names(date):
    year = str(min(max(date.year, 1992), 2018))
    return ['ESA_CCI_' + year]


def get_array(path, year):
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


def get_value(lat, lon, layer, array, array_height, array_width, path):
    year = int(layer[-4:])

    if array is None:
        array, array_width, array_height = get_array(path, year)

    lat_pos = int((array_height / 2) - (lat * (array_height / 180)))
    lon_pos = int((array_width / 2) + (lon * (array_width / 360)))

    if year < 2016:
        return int(array[lat_pos][lon_pos]), array, array_width, array_height
    else:
        return int(array[0, lat_pos, lon_pos]), array, array_width, array_height


def fill_blocks(layer, to_fetch, cell_size_degrees):
    result = []

    path = '/home/wouter/Projects/Naturalis/environment/ESACCI'

    for block_index, fetching in enumerate(to_fetch):
        if fetching == None:
            result += [None]
            continue

        block = to_fetch[block_index]['block']
        block_height, block_width = block.shape
        lat_start, lon_start = to_fetch[block_index]['lat_lon_start']

        current_file, array_height, array_width = None, 0, 0
        with tqdm(total=block_height * block_width, position=0, leave=True) as pbar:
            for row in range(block_height):
                for col in range(block_width):
                    if np.isnan(block[row, col]):
                        block[row, col], current_file, array_height, array_width = get_value(
                            lat_start - (row * cell_size_degrees),
                            lon_start + (col * cell_size_degrees),
                            layer, current_file, array_height, array_width, path)
                    pbar.update(1)
        result += [block]
        pbar.close()
    return result


def main():
    print("Call as fill_blocks(layer_name, to_fetch, cell_size_degrees)")


if __name__ == "__main__":
    main()
