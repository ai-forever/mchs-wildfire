import os
import sys
import pandas
import xarray
import requests
import datetime
import pickle
import numpy as np
from tqdm import tqdm as tqdm

DATASETS_PATH = os.environ.get('DATASETS_PATH', '../data/')

ncep_data = []
year = 2018
for var in ('air', 'uwnd', 'rhum'):
    dataset_filename = '{}/ncep/{}.{}.nc'.format(DATASETS_PATH, var, year)
    ncep_data.append(xarray.open_dataset(dataset_filename))
ncep_data = xarray.merge(ncep_data)

def extract_features(row):
    point = ncep_data.sel(
        lon=row['longitude'],
        lat=row['latitude'],
        level=1000,
        method='nearest',
    )

    p1w = point.rolling(time=7).mean()
    p2w = point.rolling(time=14).mean()
    p3w = point.rolling(time=21).mean()

    date = row['date']
    v = point.sel(time=date)
    v1w = p1w.sel(time=date)
    v2w = p2w.sel(time=date)
    v3w = p3w.sel(time=date)

    return {
        'fire_type': row['fire_type'],
        'fire_type_name': row['fire_type_name'],
        'date': row['date'],
        'temperature': v.air.values.item(0),
        'humidity': v.rhum.values.item(0),
        'uwind': v.uwnd.values.item(0),
        't1w': v1w.air.values.item(0),
        't2w': v2w.air.values.item(0),
        't3w': v3w.air.values.item(0),
        'h1w': v1w.rhum.values.item(0),
        'h2w': v2w.rhum.values.item(0),
        'h3w': v3w.rhum.values.item(0)
    }


if __name__ == '__main__':
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    df_points = pandas.read_csv(input_csv)

    df_features = []
    for i, row in tqdm(df_points.iterrows(), total=df_points.shape[0]):
        features = extract_features(row)
        df_features.append(features)
    df_features = pandas.DataFrame(df_features)

    with open('model.pickle') as fin:
        fire_classifier = pickle.load(fin)

    X = df_features.iloc[:, 3:].fillna(0)

    df_predictions = pandas.DataFrame(
        fire_classifier.predict_proba(X),
        index=df_features.index,
        columns=[
            'fire_{}_prob'.format(class_id)
            for class_id in fire_classifier.classes_
        ],
    )

    df_predictions.to_csv(output_csv)
