import math
import os
from random import randrange

import numpy as np
from tools import rastermap


def get_random_cell(raster_width, raster_height, available):
    while True:
        coords = (randrange(raster_height), randrange(raster_width))
        if available[coords]:
            return coords


def generate(presences, buffer_radius_deg=0, number=1):
    raster_max_lat = int(os.getenv("RASTER_MAX_LAT"))
    raster_min_lat = int(os.getenv("RASTER_MIN_LAT"))
    raster_max_lon = int(os.getenv("RASTER_MAX_LON"))
    raster_min_lon = int(os.getenv("RASTER_MIN_LON"))
    raster_cell_size_deg = float(os.getenv("RASTER_CELL_SIZE_DEG"))

    raster_width = int((raster_max_lon - raster_min_lon) / raster_cell_size_deg) + 1
    raster_height = int((raster_max_lat - raster_min_lat) / raster_cell_size_deg) + 1

    available = np.ones((raster_height, raster_width)) * True

    block_size = 1 + 2 * math.ceil(buffer_radius_deg / raster_cell_size_deg)

    # Sets cells that are not completely outside of the buffer zone around a presence to unavailable
    for presence in presences:
        # Back and forth to make sure the coordinate is in the middle
        lat, lon = presence
        y_center, x_center = rastermap.lat_lon_to_indices(
            *rastermap.indices_to_lat_lon(*rastermap.lat_lon_to_indices(*presence)))

        available[y_center][x_center] = False

        x_corner = rastermap.to_start_index(x_center, block_size)
        y_corner = rastermap.to_start_index(y_center, block_size)
        for x in range(x_corner, x_corner + block_size):
            for y in range(y_corner, y_corner + block_size):
                lat_cell, lon_cell = rastermap.indices_to_lat_lon(y, x)
                # check if the corner of this cell that is closest to the observation point is within the buffer radius
                if y >= 0 and y < raster_height and x >= 0 and x < raster_width and math.sqrt(
                        (abs(lat - lat_cell) - (raster_cell_size_deg / 2)) ** 2 + (
                                abs(lon - lon_cell) - (raster_cell_size_deg / 2)) ** 2) <= buffer_radius_deg:
                    available[y][x] = False

    pseudo_absences = []

    for _ in range(number):
        y, x = get_random_cell(raster_width, raster_height, available)
        available[y][x] = 5
        pseudo_absences.append(rastermap.indices_to_lat_lon(y, x))

    import matplotlib.pyplot as plt

    plt.imshow(available)
    plt.show()

    return pseudo_absences
