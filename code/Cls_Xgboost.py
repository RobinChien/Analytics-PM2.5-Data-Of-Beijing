import xgboost as xgb
import Preprocessing
import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.metrics import mean_squared_error


data = Preprocessing.Preprocessing().importLevelData()

# Train a simple linear regression model
regr = LinearRegression()
new_data = data[['season', 'DEWP', 'PRES', 'TEMP']]

X = new_data.values
y = data.level.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=42)

regr.fit(X_train, y_train)
print(regr.predict(X_test))
print(regr.score(X_test,y_test))
print("RMSE: %.2f" % math.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2)))

# Let's try XGboost algorithm to see if we can get better results
xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
traindf, testdf = train_test_split(X_train, test_size = 0.3)
xgb.fit(X_train,y_train)

predictions = xgb.predict(X_test)
print(mean_squared_error(predictions,y_test) ** 0.5)

