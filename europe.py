import os

import datetime
import itertools
import numpy as np
import pandas as pd

working_dir = "Documents/sdm_dl/data"
cell_size_deg = 1 / 6.
min_lat = 37
max_lat = 72
min_lon = -25
max_lon = 43

# Remember to set the environment variables
os.environ["RASTER_MAX_LAT"] = str(max_lat)
os.environ["RASTER_MIN_LAT"] = str(min_lat)
os.environ["RASTER_MIN_LON"] = str(min_lon)
os.environ["RASTER_MAX_LON"] = str(max_lon)
os.environ["RASTER_CELL_SIZE_DEG"] = str(cell_size_deg)
os.environ["PROJECT_ROOT"] = os.path.expanduser(f"~/{working_dir}/sdm_dl")

os.environ["BALTIC_FILE_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/BALTIC")
os.environ["GLOBE_FILE_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/GLOBE/c10g")
os.environ["ESACCI_FOLDER_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/ESACCI")
os.environ["WORLDCLIM_FOLDER_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/WORLDCLIM")
os.environ["BIOCLIM_FOLDER_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/BIOCLIM")

os.environ["RASTER_CACHE_FOLDER_PATH"] = os.path.expanduser(f"~/{working_dir}/sdm_dl/env/cache")
output_folder = os.path.expanduser(f"~/{working_dir}/sdm_dl/out")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.environ["ESACCI_FOLDER_PATH"], exist_ok=True)
# os.makedirs(os.environ["GLOBE_FILE_PATH"], exist_ok=True)

import get_environmental_layer as get_env

from layer_readers import latlon, bioclim, worldclim, esa_cci, GLOBE_elevation


def main():
    europe_left_upper = (71.02, -25.49)  # (N,E), (lat,lon)
    europe_right_lower = (37.58, 42.71)  # (N,E), (lat,lon)

    coordinate_grid = list(itertools.product(np.arange(min_lat, max_lat, cell_size_deg), np.arange(min_lon, max_lon, cell_size_deg),
                                             [datetime.datetime(2015, 6, 6), ]))
    print("len(coordinate_grid)", len(coordinate_grid))

    print(coordinate_grid[:10])

    df = pd.DataFrame(data=coordinate_grid, columns=['lat', 'lon', 'datetime'])
    df['label'] = 1

    locations = list(zip(df.lat, df.lon, df.datetime))

    # exit(0)
    layer_reader = latlon.LayerReader()
    df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 1, layer_reader)), rsuffix='_latlon')

    features = ["esa", "globe_elevation", "worldclim", "bioclim"]
    features = ["worldclim"]
    if "esa" in features:
        layer_reader = esa_cci.LayerReader()
        df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 1, layer_reader)), rsuffix='_esacci')

    if "globe_elevation" in features:
        layer_reader = GLOBE_elevation.LayerReader()
        df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 1, layer_reader)), rsuffix='_globe')

    if "worldclim" in features:
        layer_reader = worldclim.LayerReader()
        df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 1, layer_reader)), rsuffix='_worldclim')

    if "bioclim" in features:
        layer_reader = bioclim.LayerReader()
        df = df.join(pd.DataFrame.from_dict(get_env.get_blocks_as_columns(locations, 1, layer_reader)), rsuffix='_bioclim')


    df = df.drop(['lat', 'lon', 'datetime'], axis=1)

    output_file = os.path.join(output_folder, "out.npz")

    layers_ = df.drop(['label'], axis=1).to_numpy()
    print("len(coordinate_grid)", len(coordinate_grid))
    for p in range(len(coordinate_grid)):
        assert layers_[0][1] == float(coordinate_grid[0][1])
    # print(layers_[0][1], coordinate_grid[0])
    np.savez(output_file, label=df['label'].to_numpy(), columns=np.array(list(df.drop(['label'], axis=1).columns)),
             layers=layers_, meta={"cell_size_deg": cell_size_deg, "min_lat": min_lat, "max_lat": max_lat,
                                   "min_lon": min_lon, "max_lon": max_lon})

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
