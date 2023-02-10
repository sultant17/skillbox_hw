# <YOUR_IMPORTS>
import dill
import json
import pandas as pd
import os
from datetime import datetime


path = os.environ.get('PROJECT_PATH', '.')


def predict():
    # <YOUR_CODE>
    with open(f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl', 'rb') as file:
        model = dill.load(file)

    base_dir = f'{path}/data/test'

    result = None
    for file in os.listdir(base_dir):
        if file.endswith('json'):
            with open(f'{base_dir}/{file}') as cars:
                form = json.load(cars)
                df = pd.DataFrame.from_dict([form])
                y = model.predict(df)
                temp = df[['id']].copy()
                temp['predict'] = y[0]
                if result is None:
                    result = temp.copy()
                else:
                    result = pd.concat([result, temp])

    result.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
