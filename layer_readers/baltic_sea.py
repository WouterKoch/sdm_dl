#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os.path

import numpy as np
from osgeo import gdal


class LayerReader:

    ###Traditional
    def get_array_from_tif(self, _, baltic_variable):
        path = os.environ["BALTIC_FILE_PATH"]
        ds = gdal.Open(os.path.join(path, baltic_variable + '.tif'))
        band = ds.GetRasterBand(1)
        #print(np.max(band.ReadAsArray()))
        #try and extract value in one go
        return band.ReadAsArray()


    def get_value_from_array(self, lat, lon, cci_array, array_height, array_width):
        lat_pos = int((array_height / 2) - (lat * (array_height / 180)))
        lon_pos = int((array_width / 2) + (lon * (array_width / 360)))
        value = cci_array[lat_pos][lon_pos]
        #if value < -500:
        #    return np.nan
        return value
    #############################
    # Try this approach instead #
    #############################
    def get_band_from_tif(self, _,baltic_variable):
        path = os.environ["BALTIC_FILE_PATH"]
        ds = gdal.Open(os.path.join(path, baltic_variable + '.tif'))
        band = ds.GetRasterBand(1)

        cols = ds.RasterXSize
        rows = ds.RasterYSize

        transform = ds.GetGeoTransform()

        xOrigin = transform[0]
        yOrigin = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]

        array = band.ReadAsArray(0,0,cols,rows)
        return array,xOrigin,yOrigin,pixelWidth,pixelHeight

    def get_value_from_band(self, lat,lon,inp_array,xOrigin,yOrigin,pixelWidth,pixelHeight):
        col = int((lon - xOrigin) / pixelWidth)
        row = int((lat - yOrigin) / pixelHeight)
       # print(inp_array.shape,'\n',
        #      lat,'\n',
        #      lon,'\n',
        #      xOrigin,'\n',
        #      yOrigin,'\n',
        #      col,'\n',
        #      row,'\n')
        return inp_array[row,col]
    ####################

    def get_layer_names(self, time):
        filenames = os.listdir(os.environ["BALTIC_FILE_PATH"])
        suffix = '.tif'
        year = str(time.year)
        #print(year)
        layer_names = []

        for fn in sorted(filenames):
            fn = fn.replace(suffix, '')
            layer_names.append(fn)
        return [layer_name for layer_name in layer_names if year in layer_name]


    def get_layer_from_file(self, layer_name):
        filename = os.path.join(os.getenv("RASTER_CACHE_FOLDER_PATH"), 'baltic', layer_name + '.npz')
        #print(filename)
        raster_max_lat = int(os.getenv("RASTER_MAX_LAT"))
        raster_min_lat = int(os.getenv("RASTER_MIN_LAT"))
        raster_max_lon = int(os.getenv("RASTER_MAX_LON"))
        raster_min_lon = int(os.getenv("RASTER_MIN_LON"))
        raster_cell_size_deg = float(os.getenv("RASTER_CELL_SIZE_DEG"))

        layer = {}

        if os.path.exists(filename):
            layer = dict(np.load(filename, allow_pickle=True))
            #print(layer)
            #print('exists')
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
        #print(to_fetch)
        for block in to_fetch:
            if block is not None:
                break
            return [None] * len(to_fetch)

        path = os.getenv("BALTIC_FOLDER_PATH")
        #array = self.get_array_from_tif(path, layer_name)  # [-2:])
        inp_array, xOrigin, yOrigin, pixelWidth, pixelHeight = self.get_band_from_tif(path,layer_name)
        #print(xOrigin, yOrigin, pixelWidth, pixelHeight)
        #array_height, array_width = array.shape
        #print(inp_array.shape)
        #print(np.min(array),np.max(array))
        result = []

        #print(len(to_fetch))
        for block_index in range(len(to_fetch)):
            #print(block_index)
            if to_fetch[block_index] == None:
                result += [None]
                continue
            block = to_fetch[block_index]['block']
            block_height, block_width = block.shape
            #print(block.shape)
            lat_start, lon_start = to_fetch[block_index]['lat_lon_start']
            # print(lat_start_index,lon_start_index)

            for row in range(block_height):
                for col in range(block_width):
                    if np.isnan(block[row, col]):
                        # block[row, col] = self.get_value_from_array(lat_start_index - (row * cell_size_degrees),
                        #                                       lon_start_index + (col * cell_size_degrees), array,
                        #                                       array_height,
                        #                                       array_width)
                        block[row,col] = self.get_value_from_band(lat=lat_start - (row * cell_size_degrees),
                                                                  lon= lon_start + (col * cell_size_degrees),
                                                                  inp_array=inp_array,
                                                                  xOrigin=xOrigin,
                                                                  yOrigin=yOrigin,
                                                                  pixelWidth=pixelWidth,
                                                                  pixelHeight=pixelHeight)
                        #print(block[row,col])
                        # print(lat_start_index - (row * cell_size_degrees),lon_start_index + (col * cell_size_degrees), np.quantile(array,0.8),array_height,array_width)
                        # print(get_value_from_array(lat_start_index - (row * cell_size_degrees),
                        #                                       lon_start_index + (col * cell_size_degrees), array,
                        #                                       array_height,
                        #                                       array_width))
            result += [block]
        #print(result)
        return result
