from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import os

class Preprocessing():

    def importData(self, file='../res/pm25-data-of-five-chinese-cities/BeijingPM20100101_20151231.csv'):
        df = pd.read_csv(file, usecols=[1, 2, 3, 4, 5, 9, 10, 11, 12, 13], low_memory=False)
        df = df.dropna().reset_index(drop=True)
        return df

    def splitData(self, data, scaled_features):
        df = pd.concat([data[['year', 'month', 'day', 'hour', 'season', 'PM_US Post']], scaled_features[['DEWP', 'PRES', 'TEMP']]], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df.drop(['PM_US Post'] , axis=1), df['PM_US Post'], test_size=0.30, random_state=42)
        return X_train, X_test, y_train, y_test

    def normalizeData(self, data):
        scaler = StandardScaler()
        scaler.fit(data.drop(['year', 'month', 'day', 'hour', 'season', 'PM_US Post'] , axis =1))
        scaled_features = scaler.transform(data.drop(['year', 'month', 'day', 'hour', 'season', 'PM_US Post'] , axis=1))
        scaled_features = pd.DataFrame(scaled_features, columns=data.columns[6:])
        return scaled_features

    #####
    # 將Pm2.5資料匯入
    # 分割成6個區間[1, 2, 3, 4, 5, 6]
    # 1是最不嚴重，6最嚴重
    # 存入level的欄位
    #####
    def importLevelData(self, file='../res/pm25-data-of-five-chinese-cities/BeijingPM20100101_20151231.csv'):
        df = pd.read_csv(file, usecols=[1, 2, 3, 4, 5, 9, 10, 11, 12, 13], low_memory=False)
        df = df.dropna().reset_index(drop=True)
        conditions = [
            (df['PM_US Post'] <= 50),
            (df['PM_US Post'] > 50) & (df['PM_US Post']<=100),
            (df['PM_US Post'] > 100) & (df['PM_US Post']<=150),
            (df['PM_US Post'] > 151) & (df['PM_US Post']<=200),
            (df['PM_US Post'] > 201) & (df['PM_US Post']<= 300),
            (df['PM_US Post'] > 300)
            ]
        choices = [0,1,2,3,4,5]
        df['level'] = np.select(conditions, choices, default=0)
        return df

    #####
    # 將保有level欄位的Pm2.5資料集做分割
    # X_train=['DEWP', 'HUMI', 'PRES', 'TEMP'], y_train=['level']
    # X_test=['DEWP', 'HUMI', 'PRES', 'TEMP'], y_test=['level']
    #####
    def splitLevelData(self, data, scaled_features):
        df = pd.concat([data[['year', 'month', 'day', 'hour', 'season', 'PM_US Post', 'level']], scaled_features[['DEWP', 'HUMI', 'PRES', 'TEMP']]], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df.drop(['level'] , axis=1), df['level'], test_size=0.30, random_state=42)
        return X_train, X_test, y_train, y_test