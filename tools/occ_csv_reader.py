import pandas as pd

def occ_csv_reader(file, lat_col, lon_col, date_col, label_col_id):
    df_occurrence = pd.read_csv(file)
    label_cols = list(df_occurrence.columns[label_col_id])
    df_occurrence['DateTime'] = pd.to_datetime(df_occurrence["DateTime"])
    obs_list = []

    for obs in range(len(df_occurrence)):

        obs_list.append((df_occurrence[lat_col][obs], df_occurrence[lon_col][obs], df_occurrence[date_col][obs],[df_occurrence[label][obs] for label in label_cols]))

    df = pd.DataFrame.from_records(obs_list, columns=['lat', 'lon', 'datetime', 'label'])
    #return obs_list
    return df
