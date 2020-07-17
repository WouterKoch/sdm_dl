import os

raster_max_lat = int(os.getenv("RASTER_MAX_LAT"))
raster_min_lat = int(os.getenv("RASTER_MIN_LAT"))
raster_max_lon = int(os.getenv("RASTER_MAX_LON"))
raster_min_lon = int(os.getenv("RASTER_MIN_LON"))
raster_cell_size_deg = float(os.getenv("RASTER_CELL_SIZE_DEG"))


def lat_lon_to_indices(lat, lon):
    return int((raster_max_lat - lat) / raster_cell_size_deg), int((lon - raster_min_lon) / raster_cell_size_deg)


def indices_to_lat_lon(lat_index, lon_index):
    return raster_max_lat - (lat_index + .5) * raster_cell_size_deg, (
            lon_index + .5) * raster_cell_size_deg + raster_min_lon


def to_start_index(center, block_size):
    return center - int(block_size / 2)
