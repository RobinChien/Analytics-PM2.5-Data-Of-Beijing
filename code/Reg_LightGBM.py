from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import lightgbm as lgb
import pandas as pd
import numpy as np
import Preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

data = Preprocessing.Preprocessing().importData()
scaled_data = Preprocessing.Preprocessing().normalizeData(data)
X_train, X_test, y_train, y_test = Preprocessing.Preprocessing().splitData(data, scaled_data)

lgb_train = lgb.Dataset(data=X_train[['DEWP', 'HUMI', 'PRES', 'TEMP']],label=y_train)
lgb_eval = lgb.Dataset(data=X_test[['DEWP', 'HUMI', 'PRES', 'TEMP']],label=y_test)

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

gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, evals_result=evals_result, early_stopping_rounds=9)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

print('Plotting metrics recorded during training...')
ax = lgb.plot_metric(evals_result, metric='auc')
plt.show()

print('Plotting feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()