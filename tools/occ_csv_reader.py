import pandas as pd

def occ_csv_reader(file, lat_col, lon_col, date_col, label_col_id):
    df_occurrence = pd.read_csv(file)
    label_cols = list(df_occurrence.columns[label_col_id])
    df_occurrence['DateTime'] = pd.to_datetime(df_occurrence["DateTime"])
    obs_list = []
    label_list = []

    for obs in range(len(df_occurrence)):

        obs_list.append((df_occurrence[lat_col][obs], df_occurrence[lon_col][obs], df_occurrence[date_col][obs]))
        label_list.append([df_occurrence[label][obs] for label in label_cols])

    return obs_list, label_list

#example
path = '/Users/markrademaker/Projects/Naturalis/'

obs_tuple,labels=occ_csv_reader(path+'datasets/fish_occurrences.csv',
               "ShootLong","ShootLat","DateTime",label_col_id=slice(6,15))

print(obs_tuple[0])
print(labels[0])