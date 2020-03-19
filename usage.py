import os
import matplotlib.pyplot as plt
from datetime import datetime

# Remember to set the environment variables
os.environ["RASTER_MAX_LAT"] = "54"
os.environ["RASTER_MIN_LAT"] = "50"
os.environ["RASTER_MIN_LON"] = "3"
os.environ["RASTER_MAX_LON"] = "8"
os.environ["RASTER_CELL_SIZE_DEG"] = str(1 / 120)
os.environ["PROJECT_ROOT"] = "/home/wouter/Projects/Naturalis/sdm_dl"
os.environ["GLOBE_FILE_PATH"] = "/home/wouter/Projects/Naturalis/environment/GLOBE/c10g"
os.environ["ESACCI_FOLDER_PATH"] = "/home/wouter/Projects/Naturalis/environment/ESACCI"
os.environ["WORLDCLIM_FOLDER_PATH"] = "/home/wouter/Projects/Naturalis/environment/WorldClim"
os.environ["BIOCLIM_FOLDER_PATH"] = "/home/wouter/Projects/Naturalis/environment/BioClim"
os.environ["RASTER_CACHE_FOLDER_PATH"] = "/home/wouter/Projects/Naturalis/environment/sdm_dl_cache"

import get_environmental_layer as get_env

maps = get_env.get_blocks([(52.7, 5.1, datetime(2012, 1, 6))], 25, 'worldclim')

for position in maps:
    for name, map in position.items():
        plt.imshow(map)
        plt.title(name)
        plt.show()
