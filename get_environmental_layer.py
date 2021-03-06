from tqdm import tqdm
import os.path
import numpy as np

from layer_readers import bioclim as bioclim_layer_reader, GLOBE_elevation as elevation_layer_reader, \
    esa_cci as esacci_layer_reader, worldclim as worldclim_layer_reader

raster_max_lat = int(os.getenv("RASTER_MAX_LAT"))
raster_min_lat = int(os.getenv("RASTER_MIN_LAT"))
raster_max_lon = int(os.getenv("RASTER_MAX_LON"))
raster_min_lon = int(os.getenv("RASTER_MIN_LON"))
raster_cell_size_deg = float(os.getenv("RASTER_CELL_SIZE_DEG"))


def lat_lon_to_indices(lat, lon):
    return int((raster_max_lat - lat) / raster_cell_size_deg), int((lon - raster_min_lon) / raster_cell_size_deg)


def indices_to_lat_lon(lat_index, lon_index):
    return raster_max_lat - (lat_index + .5) * raster_cell_size_deg, (
            lon_index + .5) * raster_cell_size_deg + raster_min_lon


def load_block(map, lat_start, lon_start, block_size):
    return map[lat_start:lat_start + block_size, lon_start:lon_start + block_size]


def save_block(layer_map, lat_start_index, lon_start_index, block):
    block_height, block_width = block.shape
    layer_map[lat_start_index:lat_start_index + block_height, lon_start_index:lon_start_index + block_width] = block
    return layer_map


def to_start_index(center, block_size):
    return center - int(block_size / 2)


def get_blocks(occurrences, block_size, layer_reader):
    results = [{}] * len(occurrences)
    per_layer = {}

    for occ_index, occurrence in tqdm(enumerate(occurrences)):
        layer_names = layer_reader.get_layer_names(occurrence[2])
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
            lat_index, lon_index = lat_lon_to_indices(lat, lon)
            lat, lon = indices_to_lat_lon(to_start_index(lat_index, block_size), to_start_index(lon_index, block_size))

            results[occ_index][layer_name] = load_block(layer['map'], to_start_index(lat_index, block_size),
                                                        to_start_index(lon_index, block_size), block_size)

            if np.isnan(np.sum(results[occ_index][layer_name])):
                results[occ_index][layer_name][
                    results[occ_index][layer_name] == layer['metadata']['null_value']] = np.nan
                incomplete_ids += [occ_index]
                incomplete_blocks += [{'block': results[occ_index][layer_name], 'lat_lon_start': (lat, lon)}]

        completed_blocks = layer_reader.fill_blocks(layer_name, incomplete_blocks,
                                                    layer['metadata']['cell_size_degrees'])

        for block_index, completed_block in enumerate(completed_blocks):
            occ_index = incomplete_ids[block_index]

            if layer['metadata']['data_type'] != 'categorical':
                layer['metadata']['normalization_range'] = (
                    min(layer['metadata']['normalization_range'][0], np.nanmin(completed_block)),
                    max(layer['metadata']['normalization_range'][1], np.nanmax(completed_block)))

            completed_block[np.isnan(completed_block)] = layer['metadata']['null_value']
            results[occ_index][layer_name] = completed_block

            lat, lon, _ = layer_occurrences[occ_index]
            lat_index, lon_index = lat_lon_to_indices(lat, lon)
            layer['map'] = save_block(layer['map'], to_start_index(lat_index, block_size),
                                      to_start_index(lon_index, block_size),
                                      completed_block)
        os.makedirs(os.path.dirname(layer['filename']), exist_ok=True)
        np.savez(layer['filename'], map=layer['map'], metadata=layer['metadata'])

        if layer['metadata']['data_type'] != 'categorical':
            min_, max_ = layer['metadata']['normalization_range']

            for result in results:
                if layer_name in result:
                    result[layer_name] = (result[layer_name] - ((min_ + max_) / 2)) / ((max_ - min_) / 2)
                result[layer_name][result[layer_name] < -1] = np.nan
        else:
            categories = layer['metadata']['data_categories']

            for result in results:
                if layer_name in result:
                    for category in categories:
                        result[layer_name + '_' + str(category)] = [list(map(lambda x: int(x == category), row)) for row
                                                                    in result[layer_name]]

    return results


if __name__ == "__main__":
    print("Call as get_blocks(positions, block_size, dataset)")
