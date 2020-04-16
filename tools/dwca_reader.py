import zipfile
import pandas as pd
from datetime import datetime

def zip_to_presences(filename):
    pd.low_memory = False
    df_occurrence = pd.read_csv(zipfile.ZipFile(filename).open('occurrence.txt'), sep='\t',
                                     error_bad_lines=False) \
        .dropna(subset=['species']) \
        .dropna(subset=['decimalLatitude']) \
        .dropna(subset=['decimalLongitude']) \
        .dropna(subset=['year']) \
        .dropna(subset=['month']) \
        .dropna(subset=['day'])

    df_occurrence['datetime'] = df_occurrence.apply(lambda x: datetime(int(x['year']), int(x['month']), int(x['day'])), axis=1)
    subset = df_occurrence[['decimalLatitude', 'decimalLongitude', 'datetime']]
    return [tuple(x) for x in subset.to_numpy()]
