from tqdm import tqdm
import os.path
import numpy as np

import get_elevation_GLOBE as elevation_layer_reader
import get_bioclim_data as bioclim_layer_reader
import get_worldclim_data as worldclim_layer_reader
import get_esa_cci as esacci_layer_reader


def lat_lon_to_indices(lat, lon, cell_size_degrees):
    return int((72 - lat) / cell_size_degrees), int((lon - 4) / cell_size_degrees)


def indices_to_lat_lon(lat_index, lon_index, cell_size_degrees):
    return 72 - (lat_index + .5) * cell_size_degrees, (lon_index + .5) * cell_size_degrees + 4


def get_layer_from_file(output_file, dataset):
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    if os.path.exists(output_file):
        return np.load(output_file)
    else:
        metadata = {'lat_NW_cell_center': 72, 'lon_NW_cell_center': 4, 'cell_size_degrees': 1 / 120}

        if dataset == 'esacci':
            metadata['data_type'] = 'categorical'
            metadata['data_categories'] = [10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110,
                                           120, 121, 122, 130, 140, 150,
                                           151, 152, 153, 160, 170, 180, 190, 200, 201, 202, 210, 220]
        else:
            metadata['data_type'] = 'numerical'
            metadata['normalization_range'] = (99999999998, -99999999998)
        metadata['null_value'] = -99999999999

        return {'map': np.ones((1800, 3800)) * np.nan,
                'metadata': metadata}


def get_layer_reader(layer):
    if layer == 'elevation':
        return elevation_layer_reader
    elif layer == 'bioclim':
        return bioclim_layer_reader
    elif layer == 'worldclim':
        return worldclim_layer_reader
    elif layer == 'esacci':
        return esacci_layer_reader


def get_layer_names(dataset, date):
    if dataset == 'bioclim' or dataset == 'worldclim' or dataset == 'esacci':
        layer_reader = get_layer_reader(dataset)
        return layer_reader.get_layer_names(date)
    else:  # lat, lon, elevation
        return [dataset]


def get_layer(dataset, layer_name, cache_dir):
    layer_object = {}

    filename = os.path.join(cache_dir, dataset, layer_name + '.npz')
    layer = get_layer_from_file(filename, dataset)

    layer_object['map'] = layer['map']
    layer_object['filename'] = filename

    if isinstance(layer['metadata'], dict):
        layer_object['metadata'] = layer['metadata']
    else:
        layer_object['metadata'] = layer['metadata'].item()

    return layer_object


def load_block(map, lat_start, lon_start, block_size):
    return map[lat_start:lat_start + block_size, lon_start:lon_start + block_size]


def save_block(layer_map, lat_start_index, lon_start_index, block):
    block_height, block_width = block.shape
    layer_map[lat_start_index:lat_start_index + block_height, lon_start_index:lon_start_index + block_width] = block
    return layer_map


def to_start_index(center, block_size):
    return center - int(block_size / 2)


def get_blocks(occurrences, block_size, dataset, cache_dir):
    layer_reader = get_layer_reader(dataset)
    results = [{}] * len(occurrences)
    per_layer = {}

    for occ_index, occurrence in tqdm(enumerate(occurrences)):
        layer_names = get_layer_names(dataset, occurrence[2])
        for layer_name in layer_names:
            if layer_name not in per_layer:
                per_layer[layer_name] = {}
            per_layer[layer_name][occ_index] = occurrence


    for layer_name, layer_occurrences in per_layer.items():
        layer = get_layer(dataset, layer_name, cache_dir)
        incomplete_blocks = []
        incomplete_ids = []
        for occ_index, occurrence in layer_occurrences.items():
            lat, lon, _ = occurrence
            lat_index, lon_index = lat_lon_to_indices(lat, lon, layer['metadata']['cell_size_degrees'])
            lat, lon = indices_to_lat_lon(to_start_index(lat_index, block_size), to_start_index(lon_index, block_size), layer['metadata']['cell_size_degrees'])

            results[occ_index][layer_name] = load_block(layer['map'], to_start_index(lat_index, block_size), to_start_index(lon_index, block_size), block_size)

            if np.isnan(np.sum(results[occ_index][layer_name])):
                results[occ_index][layer_name][
                    results[occ_index][layer_name] == layer['metadata']['null_value']] = np.nan
                incomplete_ids += [occ_index]
                incomplete_blocks += [{'block': results[occ_index][layer_name], 'lat_lon_start': (lat, lon)}]


        completed_blocks = layer_reader.fill_blocks(layer_name, incomplete_blocks, layer['metadata']['cell_size_degrees'])

        for block_index, completed_block in enumerate(completed_blocks):
            occ_index = incomplete_ids[block_index]

            if layer['metadata']['data_type'] != 'categorical':
                layer['metadata']['normalization_range'] = (min(layer['metadata']['normalization_range'][0], np.nanmin(completed_block)), max(layer['metadata']['normalization_range'][1], np.nanmax(completed_block)))

            completed_block[np.isnan(completed_block)] = layer['metadata']['null_value']
            results[occ_index][layer_name] = completed_block

            lat, lon, _ = layer_occurrences[occ_index]
            lat_index, lon_index = lat_lon_to_indices(lat, lon, layer['metadata']['cell_size_degrees'])
            layer['map'] = save_block(layer['map'], to_start_index(lat_index, block_size),
                                      to_start_index(lon_index, block_size),
                                      completed_block)
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
                        result[layer_name + '_' + str(category)] = [list(map(lambda x: int(x == category), row)) for row in result[layer_name]]

    return results


if __name__ == "__main__":
    print("Call as get_blocks(positions, block_size, dataset, cache_dir)")
