import os
from random import randrange

import numpy as np
from tools import rastermap


def get_random_cell(raster_width, raster_height, available):
    while True:
        coords = (randrange(raster_width), randrange(raster_height))
        if available[coords]:
            return coords


def generate(presences, min_distance=1, number=1):
    raster_max_lat = int(os.getenv("RASTER_MAX_LAT"))
    raster_min_lat = int(os.getenv("RASTER_MIN_LAT"))
    raster_max_lon = int(os.getenv("RASTER_MAX_LON"))
    raster_min_lon = int(os.getenv("RASTER_MIN_LON"))
    raster_cell_size_deg = float(os.getenv("RASTER_CELL_SIZE_DEG"))

    raster_width = int((raster_max_lon - raster_min_lon) / raster_cell_size_deg) + 1
    raster_height = int((raster_max_lat - raster_min_lat) / raster_cell_size_deg) + 1

    available = np.ones((raster_width, raster_height)) * True

    block_size = 1 + 2 * min_distance

    for presence in presences:
        y, x = rastermap.lat_lon_to_indices(presence)
        x_corner = rastermap.to_start_index(x, block_size)
        y_corner = rastermap.to_start_index(y, block_size)
        for x in range(x_corner, x_corner + block_size):
            for y in range(y_corner, y_corner + block_size):
                available[y][x] = False

    pseudo_absences = []

    for _ in range(number):
        x, y = get_random_cell(raster_width, raster_height, available)
        pseudo_absences.append(rastermap.indices_to_lat_lon(y, x))

    return pseudo_absences
