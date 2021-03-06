import os
import matplotlib.pyplot as plt
from datetime import datetime

user= 'wouter' #'markrademaker'

# Remember to set the environment variables
os.environ["RASTER_MAX_LAT"] = "54"
os.environ["RASTER_MIN_LAT"] = "50"
os.environ["RASTER_MIN_LON"] = "3"
os.environ["RASTER_MAX_LON"] = "8"
os.environ["RASTER_CELL_SIZE_DEG"] = str(1 / 120)
os.environ["PROJECT_ROOT"] = "/home/{}/Projects/Naturalis/sdm_dl".format(user)

os.environ["BALTIC_FILE_PATH"] = "/home/{}/Projects/Naturalis/environment/BALTIC".format(user)
os.environ["GLOBE_FILE_PATH"] = "/home/{}/Projects/Naturalis/environment/GLOBE/c10g".format(user)
os.environ["ESACCI_FOLDER_PATH"] = "/home/{}/Projects/Naturalis/environment/ESACCI".format(user)
os.environ["WORLDCLIM_FOLDER_PATH"] = "/home/{}/Projects/Naturalis/environment/WorldClim".format(user)
os.environ["BIOCLIM_FOLDER_PATH"] = "/home/{}/Projects/Naturalis/environment/BioClim".format(user)

os.environ["RASTER_CACHE_FOLDER_PATH"] = "/home/{}/Projects/Naturalis/environment/sdm_dl_cache".format(user)

import get_environmental_layer as get_env

from layer_readers import worldclim as layer_reader
maps = get_env.get_blocks([(52.7, 5.1, datetime(2012, 1, 6))], 25, layer_reader)

for position in maps:
    for name, map in position.items():
        plt.imshow(map)
        plt.title(name)
        plt.show()
