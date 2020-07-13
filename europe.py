import datetime
import itertools
import os

import numpy as np
import pandas as pd

working_dir = "Documents/sdm_dl/data"

# Remember to set the environment variables
os.environ["RASTER_MAX_LAT"] = "72"
os.environ["RASTER_MIN_LAT"] = "37"
os.environ["RASTER_MIN_LON"] = "-25"
os.environ["RASTER_MAX_LON"] = "43"
os.environ["RASTER_CELL_SIZE_DEG"] = "1"
os.environ["PROJECT_ROOT"] = os.path.expanduser(f"~/{working_dir}/sdm_dl")

os.environ["BALTIC_FILE_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/BALTIC")
os.environ["GLOBE_FILE_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/GLOBE")
os.environ["ESACCI_FOLDER_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/ESACCI")
os.environ["WORLDCLIM_FOLDER_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/WORLDCLIM")
os.environ["BIOCLIM_FOLDER_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/BIOCLIM")

os.environ["RASTER_CACHE_FOLDER_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/cache")
output_folder = os.path.expanduser(f"~/{working_dir}/sdm_dl/out")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.environ["ESACCI_FOLDER_PATH"], exist_ok=True)
os.makedirs(os.environ["GLOBE_FILE_PATH"], exist_ok=True)

import get_environmental_layer as get_env

from layer_readers import bioclim
from layer_readers import latlon


def main():
    europe_left_upper = (71.02, -25.49)  # (N,E), (lat,lon)
    europe_right_lower = (37.58, 42.71)  # (N,E), (lat,lon)

    coordinate_grid = list(itertools.product(np.arange(37, 71, 1), np.arange(-25, 43, 1), [datetime.datetime(2016, 6, 6), ]))
    print("len(coordinate_grid)", len(coordinate_grid))

    print(coordinate_grid[:10])

    df = pd.DataFrame(data=coordinate_grid, columns=['lat', 'lon', 'datetime'])
    df['label'] = 1

    locations = list(zip(df.lat, df.lon, df.datetime))

    # exit(0)

    # layer_reader = esa_cci.LayerReader()
    # df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 3, layer_reader)), rsuffix='_esacci')

    # layer_reader = GLOBE_elevation.LayerReader()
    # df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 3, layer_reader)), rsuffix='_globe')

    layer_reader = bioclim.LayerReader()
    df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 1, layer_reader)), rsuffix='_bioclim')

    # layer_reader = worldclim.LayerReader()
    # df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 3, layer_reader)), rsuffix='_worldclim')

    layer_reader = latlon.LayerReader()
    df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 1, layer_reader)), rsuffix='_latlon')

    df = df.drop(['lat', 'lon', 'datetime'], axis=1)

    output_file = os.path.join(output_folder, str(int(datetime.datetime.now().timestamp() * 100)) + '.npz')
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


if __name__ == '__main__':
    main()
