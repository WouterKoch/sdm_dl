import os
import matplotlib.pyplot as plt
import pandas as pd

# user = 'Users/markrademaker'
user = 'home/wouter'

# Remember to set the environment variables
os.environ["RASTER_MAX_LAT"] = "72"
os.environ["RASTER_MIN_LAT"] = "50"
os.environ["RASTER_MIN_LON"] = "3"
os.environ["RASTER_MAX_LON"] = "33"
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

from tools import dwca_reader

presences = dwca_reader.zip_to_presences('/home/wouter/Projects/Naturalis/datasets/GBIF_sphagnum_2019-09-30.zip')
df = pd.DataFrame(presences, columns=['lat', 'lon', 'datetime'])
df['label'] = 1

pseudo_absences = pseudoabsence.generate(presences, .5, 1, 500)
df_pseudo_absences = pd.DataFrame(pseudo_absences, columns=['lat', 'lon', 'datetime'])
df_pseudo_absences['label'] = 0
df = df.append(df_pseudo_absences)

locations = list(zip(df.lat, df.lon, df.datetime))

from layer_readers import GLOBE_elevation as layer_reader
df['elevation'] = get_env.get_blocks(locations, 30, layer_reader)

# for position in maps:
#     for name, map in position.items():
#         fig, ax = plt.subplots()
#         im = ax.imshow(map, cmap=plt.get_cmap('viridis'))
#         # print(map)
#         plt.imshow(map)
#         plt.title(name)
#         plt.show()
