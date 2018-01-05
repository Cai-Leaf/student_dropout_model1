import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt
import time
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('../feature_selection/feature_final.csv', header=0)
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)

# lightgbm
clf = LGBMClassifier(num_leaves=6, learning_rate=0.05, max_depth=6, n_estimators=200, subsample=1,
                     colsample_bytree=1, min_child_weight=1, reg_alpha=1e-5*3,  reg_lambda=1e-5*3)

# 定义参数
# param = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500]}
# param = {'num_leaves': [6, 8, 10, 20, 30],
#          'max_depth': [6, 8, 10, 20, 30],
#          'min_child_weight': [1, 2, 3, 4, 5]}
# param = {'min_split_gain': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
# param = {'subsample': [1, 0.9, 0.8, 0.7, 0.6],
#          'colsample_bytree': [1, 0.9, 0.8, 0.7, 0.6]}
# param = {'reg_alpha': [1e-5*1, 1e-5*2, 1e-5*3, 1e-5*4, 1e-5*5],
#          'reg_lambda': [1e-5*1, 1e-5*2, 1e-5*3, 1e-5*4, 1e-5*5]}
param = {'learning_rate': [0.05, 0.015, 0.025],
         'n_estimators': [100, 150, 200]}
# 交叉验证
gsearch = GridSearchCV(estimator=clf, param_grid=param, scoring='roc_auc', cv=5)
gsearch.fit(data, data_y)
# print(gsearch.cv_results_)
print(gsearch.best_params_)
print(gsearch.best_score_)



