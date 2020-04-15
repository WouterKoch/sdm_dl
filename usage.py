import os
import matplotlib.pyplot as plt
from datetime import datetime

# user = 'Users/markrademaker'
user = 'home/wouter'

# Remember to set the environment variables
os.environ["RASTER_MAX_LAT"] = "66"
os.environ["RASTER_MIN_LAT"] = "58"
os.environ["RASTER_MIN_LON"] = "9"
os.environ["RASTER_MAX_LON"] = "11"
os.environ["RASTER_CELL_SIZE_DEG"] = str(1 / 120)
os.environ["PROJECT_ROOT"] = "/{}/Projects/Naturalis/sdm_dl".format(user)

os.environ["BALTIC_FILE_PATH"] = "/{}/Projects/Naturalis/environment/BALTIC".format(user)
os.environ["GLOBE_FILE_PATH"] = "/{}/Projects/Naturalis/environment/GLOBE/c10g".format(user)
os.environ["ESACCI_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/ESACCI".format(user)
os.environ["WORLDCLIM_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/WorldClim".format(user)
os.environ["BIOCLIM_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/BioClim".format(user)

os.environ["RASTER_CACHE_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/sdm_dl_cache".format(user)

from tools import pseudoabsence
import get_environmental_layer as get_env
from layer_readers import GLOBE_elevation as layer_reader

presences = [(63.3, 10, datetime(1991, 1, 6))]

pseudo_absences = pseudoabsence.generate(presences, .5, 1, 500)
maps = get_env.get_blocks(presences, 200, layer_reader)

for position in maps:
    for name, map in position.items():
        fig, ax = plt.subplots()
        im = ax.imshow(map, cmap=plt.get_cmap('viridis'))
        # print(map)
        # plt.imshow(map)
        plt.title(name)
        plt.show()
