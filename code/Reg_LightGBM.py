from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score,confusion_matrix

import lightgbm as lgb
import pandas as pd
import numpy as np
import Preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

data = Preprocessing.Preprocessing().importLevelData()
scaled_data = Preprocessing.Preprocessing().normalizeData(data)
X_train, X_test, y_train, y_test = Preprocessing.Preprocessing().splitLevelData(data, scaled_data)

X_train = X_train.drop(['year', 'month', 'day', 'hour', 'PM_US Post'], axis=1)
X_test = X_test.drop(['year', 'month', 'day', 'hour', 'PM_US Post'], axis=1)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 100,
    'learning_rate': 0.05,
    'num_iterations': 10000,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 9,
    'verbose': 1,
    'max_bin': 100
    }

evals_result = {}

gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=9)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print("y_pred:", y_pred)
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
