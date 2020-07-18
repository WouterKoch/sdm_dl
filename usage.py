import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

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
os.environ["GLOBE_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/GLOBE".format(user)
os.environ["GEBCO_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/GEBCO".format(user)
os.environ["ESACCI_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/ESACCI".format(user)
os.environ["WORLDCLIM_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/WorldClim".format(user)
os.environ["BIOCLIM_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/BioClim".format(user)

os.environ["RASTER_CACHE_FOLDER_PATH"] = "/{}/Projects/Naturalis/environment/sdm_dl_cache".format(user)
output_folder = "/{}/Projects/Naturalis/datasets/sdm_dl".format(user)

from tools import pseudoabsence
import get_environmental_layer as get_env

from layer_readers import esa_cci
from layer_readers import GLOBE_elevation
from layer_readers import GEBCO_elevation
from layer_readers import bioclim
from layer_readers import worldclim
from layer_readers import latlon

from tools import dwca_reader

presences = dwca_reader.zip_to_presences('/home/wouter/Projects/Naturalis/datasets/GBIF_rosechafers_2019-09-24.zip')
df = pd.DataFrame(presences, columns=['lat', 'lon', 'datetime'])
df['label'] = 1

pseudo_absences = pseudoabsence.generate(presences, .25, .75, len(df))
df_pseudo_absences = pd.DataFrame(pseudo_absences, columns=['lat', 'lon', 'datetime'])
df_pseudo_absences['label'] = 0
df = df.append(df_pseudo_absences)

locations = list(zip(df.lat, df.lon, df.datetime))
#
# layer_reader = esa_cci.LayerReader()
# df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 7, layer_reader)), rsuffix='_esacci')

layer_reader = GEBCO_elevation.LayerReader()
df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 7, layer_reader)), rsuffix='_gebco')
#
# layer_reader = bioclim.LayerReader()
# df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 7, layer_reader)), rsuffix='_bioclim')
#
# layer_reader = worldclim.LayerReader()
# df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 7, layer_reader)), rsuffix='_worldclim')
#
# layer_reader = latlon.LayerReader()
# df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 7, layer_reader)), rsuffix='_latlon')

df = df.drop(['lat', 'lon', 'datetime'], axis=1)

output_file = os.path.join(output_folder, str(int(datetime.datetime.now().timestamp() * 100)) + '.npz');
np.savez(output_file, label=df['label'].to_numpy(), columns=np.array(list(df.drop(['label'], axis=1).columns)),
         layers=df.drop(['label'], axis=1).to_numpy())

# for position in maps:
#     for name, map in position.items():
#         fig, ax = plt.subplots()
#         im = ax.imshow(map, cmap=plt.get_cmap('viridis'))
#         # print(map)
#         plt.imshow(map)
#         plt.title(name)
#         plt.show()
