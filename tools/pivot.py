import zipfile

import numpy as np
import pandas as pd


def read_dwca(dwca_file):
    '''Returns a dataframe with only lat, lon, and species of all observations in a DwCA zip file'''
    return pd.read_csv(zipfile.ZipFile(dwca_file).open('occurrence.txt'), sep='\t', error_bad_lines=False,
                       usecols=['decimalLatitude', 'decimalLongitude', 'species'])


def get_pivot(path):
    '''Based on a DwCA file, returns a dataframe with a presence (1 or 0) for each species (cols) per grid cell (rows)'''
    occ = read_dwca(path).dropna()
    occ['decimalLatitude'] = np.floor(occ['decimalLatitude'] * 120) / 120 + (1 / 240)
    occ['decimalLongitude'] = np.floor(occ['decimalLongitude'] * 120) / 120 + (1 / 240)
    occ.drop_duplicates(inplace=True)
    occ['ones'] = 1
    occ['ones'] = occ['ones'].astype('uint8')

    # Add adjacent occurrences? (% of neighbors with species)
    # Or a radius?

    return occ.pivot(index=['decimalLatitude', 'decimalLongitude'], columns='species', values='ones')


def store_per_latitude(dataframe, folder, from_degrees, stepsize):
    '''Stores a dataframe containing a decimalLatitude column in separate files in the specified folder'''
    while from_degrees < 90:
        selection = dataframe.query('decimalLatitude > ' + str(from_degrees) + ' & decimalLatitude < ' + str(from_degrees + stepsize))
        if len(selection):
            selection.fillna(0).astype('int32').to_csv(folder + '/' + str(from_degrees) + '.csv')
        from_degrees += stepsize


if __name__ == '__main__':
    print('Usage: \n'
          'store_per_latitude(dataframe, folder, from_degrees, stepsize)\n'
          'get_pivot(path)')
    exit(0)

    # Example usage:

    # file = '/home/wouter/Projects/Naturalis/datasets/GBIF_all_ArtsObs_and_iNaturalist_pics_2020-04-07.zip'
    file = '/home/wouter/Projects/Naturalis/datasets/GBIF_ALL_bombus_2019-11-28.zip'
    #  file = '/home/wouter/Projects/Naturalis/datasets/GBIF_ALL_plantae_2019-11-25.zip'

    degrees = 55.0
    stepsize = .5
    folder = 'Bombus'

    store_per_latitude(get_pivot(file), folder, degrees, stepsize)
