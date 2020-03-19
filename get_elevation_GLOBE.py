import os
from tqdm import tqdm
import numpy as np
from struct import unpack


def get_value(lat, lon, current_file):
    if current_file is None:
        current_file = open(os.getenv("GLOBE_FILE_PATH"), "rb")

    get_row = (90 * 60 * 2) - int(lat * 60 * 2)
    get_col = int(lon * 60 * 2)
    n_cols = 10800
    n_bytes = 2

    current_file.seek(get_row * n_bytes * n_cols + get_col * n_bytes, 0)
    value = unpack('h', current_file.read(n_bytes))[0]
    if value == -500:
        value = np.nan

    return value, current_file


def fill_blocks(_, to_fetch, cell_size_degrees):
    result = []

    for block_index in range(len(to_fetch)):
        if to_fetch[block_index] == None:
            result += [None]
            continue
        block = to_fetch[block_index]['block']
        block_height, block_width = block.shape
        lat_start, lon_start = to_fetch[block_index]['lat_lon_start']

        current_file = None
        with tqdm(total=block_height * block_width, position=0, leave=True) as pbar:
            for row in range(block_height):
                for col in range(block_width):
                    if np.isnan(block[row, col]):
                        block[row, col], current_file = get_value(lat_start - (row * cell_size_degrees),
                                                                  lon_start + (col * cell_size_degrees),
                                                                  current_file)
                    pbar.update(1)
        result += [block]
        pbar.close()
    return result