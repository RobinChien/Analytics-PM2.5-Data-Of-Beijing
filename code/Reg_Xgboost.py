from sklearn.metrics import accuracy_score

import Preprocessing
import pandas
import xgboost as xgb

data = Preprocessing.Preprocessing().importData()
scaled_data = Preprocessing.Preprocessing().normalizeData(data)
X_train, X_test, y_train, y_test = Preprocessing.Preprocessing().splitData(scaled_data, data)
# param = {
        
#         }

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc")
xgb_model.fit(X_train[:-4], y_train[:-4], early_stopping_rounds=10, eval_set=[(X_test, y_test)])
# y_pred = xgb_model.predict(X_test[:-4])
# accuracy = accuracy_score(y_test, y_pred)
# print('The accurancy of XGBoost Classifier on testing set:', xgb_model.best_score)
# print("best score: {0}, best iteration: {1}, best ntree limit {2}".format(xgb_model.best_score, xgb_model.best_iteration, xgb_model.best_ntree_limit))