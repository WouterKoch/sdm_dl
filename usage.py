import os
import matplotlib.pyplot as plt
from datetime import datetime

user= 'Users/markrademaker'#'home/wouter'

# Remember to set the environment variables
os.environ["RASTER_MAX_LAT"] = "60"
os.environ["RASTER_MIN_LAT"] = "52"
os.environ["RASTER_MIN_LON"] = "9"
os.environ["RASTER_MAX_LON"] = "24"
os.environ["RASTER_CELL_SIZE_DEG"] = str(1 / 120)
os.environ["PROJECT_ROOT"] = "{}/Projects/Naturalis/sdm_dl".format(user)

os.environ["BALTIC_FILE_PATH"] = "/{}/Projects/Naturalis/environment/BALTIC".format(user)
os.environ["GLOBE_FILE_PATH"] = "/{}/Projects/Naturalis/environment/GLOBE/c10g".format(user)
os.environ["ESACCI_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/ESACCI".format(user)
os.environ["WORLDCLIM_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/WorldClim".format(user)
os.environ["BIOCLIM_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/BioClim".format(user)

os.environ["RASTER_CACHE_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/sdm_dl_cache".format(user)

import get_environmental_layer as get_env

from layer_readers import baltic_sea as layer_reader
maps = get_env.get_blocks([(56.1, 12, datetime(1991, 1, 6))], 25, layer_reader)

for position in maps:
    for name, map in position.items():
        fig,ax = plt.subplots()
        im = ax.imshow(map,cmap=plt.get_cmap('hot'))
        print(map)
        fig.colorbar(im)
        #plt.imshow(map)
        plt.title(name)
        plt.show()
