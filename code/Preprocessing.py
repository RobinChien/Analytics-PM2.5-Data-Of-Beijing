from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import os

class Preprocessing():

    def importData(self, file='../res/pm25-data-of-five-chinese-cities/BeijingPM20100101_20151231.csv'):
        df = pd.read_csv(file, usecols=[1, 2, 3, 4, 5, 9, 10, 11, 12, 13], low_memory=False)
        df = df.dropna().reset_index(drop=True)
        return df

    def splitData(self, scaled_features, data):
        df = pd.concat([data[['year', 'month', 'day', 'hour', 'season', 'PM_US Post']], scaled_features], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df.drop(['PM_US Post'] , axis=1), df['PM_US Post'], test_size=0.30, random_state=42)
        return X_train, X_test, y_train, y_test

    def normalizeData(self, data):
        scaler = StandardScaler()
        scaler.fit(data.drop(['year', 'month', 'day', 'hour', 'season', 'PM_US Post'] , axis =1))
        scaled_features = scaler.transform(data.drop(['year', 'month', 'day', 'hour', 'season', 'PM_US Post'] , axis=1))
        scaled_features = pd.DataFrame(scaled_features, columns=data.columns[6:])
        return scaled_features