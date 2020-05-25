from tqdm import tqdm
import os.path
import numpy as np
from tools import rastermap

raster_max_lat = int(os.getenv("RASTER_MAX_LAT"))
raster_min_lat = int(os.getenv("RASTER_MIN_LAT"))
raster_max_lon = int(os.getenv("RASTER_MAX_LON"))
raster_min_lon = int(os.getenv("RASTER_MIN_LON"))
raster_cell_size_deg = float(os.getenv("RASTER_CELL_SIZE_DEG"))


def load_block(map, lat_start, lon_start, block_size):
    return map[lat_start:lat_start + block_size, lon_start:lon_start + block_size]


def save_block(layer_map, lat_start_index, lon_start_index, block):
    block_height, block_width = block.shape
    layer_map[lat_start_index:lat_start_index + block_height, lon_start_index:lon_start_index + block_width] = block
    return layer_map


def get_blocks(occurrences, block_size, layer_reader):
    results = [{} for _ in range(len(occurrences))]
    per_layer = {}

    for occ_index, occurrence in tqdm(enumerate(occurrences)):
        date = occurrence[2]
        layer_names = layer_reader.get_layer_names(date)
        for layer_name in layer_names:
            if layer_name not in per_layer:
                per_layer[layer_name] = {}
            per_layer[layer_name][occ_index] = occurrence

    for layer_name, layer_occurrences in per_layer.items():
        layer = layer_reader.get_layer_from_file(layer_name)
        incomplete_blocks = []
        incomplete_ids = []

        for occ_index, occurrence in layer_occurrences.items():
            lat, lon, _ = occurrence
            lat_index, lon_index = rastermap.lat_lon_to_indices(lat, lon)
            lat, lon = rastermap.indices_to_lat_lon(rastermap.to_start_index(lat_index, block_size),
                                                    rastermap.to_start_index(lon_index, block_size))

            results[occ_index][layer_name] = load_block(layer['map'], rastermap.to_start_index(lat_index, block_size),
                                                        rastermap.to_start_index(lon_index, block_size), block_size)

            # if some values were not present in the cache, remember this block as being incomplete
            if np.isnan(np.sum(results[occ_index][layer_name])):
                results[occ_index][layer_name][
                    results[occ_index][layer_name] == layer['metadata']['null_value']] = np.nan
                incomplete_ids += [occ_index]
                incomplete_blocks += [{'block': results[occ_index][layer_name], 'lat_lon_start': (lat, lon)}]
        # fill the incomplete blocks, if any

        completed_blocks = layer_reader.fill_blocks(layer_name, incomplete_blocks,
                                                    layer['metadata']['cell_size_degrees'])

        # add the previously incomplete blocks, update metadata such as the min-max range, and write to file
        for block_index, completed_block in enumerate(completed_blocks):
            occ_index = incomplete_ids[block_index]

            if layer['metadata']['data_type'] != 'categorical':
                layer['metadata']['normalization_range'] = (
                    min(layer['metadata']['normalization_range'][0], np.nanmin(completed_block)),
                    max(layer['metadata']['normalization_range'][1], np.nanmax(completed_block)))

            # completed_block[np.isnan(completed_block)] = layer['metadata']['null_value']

            results[occ_index][layer_name] = completed_block

            results[occ_index][layer_name] = completed_block

            lat, lon, _ = layer_occurrences[occ_index]
            lat_index, lon_index = rastermap.lat_lon_to_indices(lat, lon)
            layer['map'] = save_block(layer['map'], rastermap.to_start_index(lat_index, block_size),
                                      rastermap.to_start_index(lon_index, block_size),
                                      completed_block)
        os.makedirs(os.path.dirname(layer['filename']), exist_ok=True)
        np.savez(layer['filename'], map=layer['map'], metadata=layer['metadata'])

        if layer['metadata']['data_type'] != 'categorical':
            min_, max_ = layer['metadata']['normalization_range']

            for result in results:
                if layer_name in result:
                    result[layer_name] = (result[layer_name] - ((min_ + max_) / 2)) / ((max_ - min_) / 2)
                result[layer_name][result[layer_name] < -1] = None
                # This will give a warning "RuntimeWarning: invalid value encountered in less"
                # that's because there are missing values in the output, although that is the whole point
                # This happens for both None and np.nan, while these are accepted as missing value notation.
                # E.g. https://github.com/Unidata/netcdf4-python/issues/766
                # See also https://stackoverflow.com/questions/28654325/what-is-pythons-equivalent-of-rs-na

        else:
            categories = layer['metadata']['data_categories']

            for result in results:
                if layer_name in result:
                    # split the category values into a new element for each category
                    for category in categories:
                        result[layer['metadata']['data_labels'][category]] = [
                            list(map(lambda x: int(x == category), row)) for row
                            in result[layer_name]]
                    # we don't need the original category values, since they were just split
                    del result[layer_name]

    return results


def get_blocks_as_columns(occurrences, block_size, layer_reader):
    rows = get_blocks(occurrences, block_size, layer_reader)
    columns = {}
    for row_index, row in enumerate(rows):
        for col_label, col in row.items():
            if col_label not in columns:
                columns[col_label] = [None for _ in range(len(rows))]
            columns[col_label][row_index] = np.array(col)

    return columns


if __name__ == "__main__":
    print("Call as get_blocks(positions, block_size, dataset)")
