import abc
import os

import datetime
from typing import List


class AbstractLayerReader(object):

    @abc.abstractmethod
    def get_layer_names(self, date: datetime.datetime) -> List[str]:
        pass


def get_raster():
    raster_max_lat = int(os.getenv("RASTER_MAX_LAT"))
    raster_min_lat = int(os.getenv("RASTER_MIN_LAT"))
    raster_max_lon = int(os.getenv("RASTER_MAX_LON"))
    raster_min_lon = int(os.getenv("RASTER_MIN_LON"))
    raster_cell_size_deg = float(os.getenv("RASTER_CELL_SIZE_DEG"))
    return raster_cell_size_deg, raster_max_lat, raster_max_lon, raster_min_lat, raster_min_lon
