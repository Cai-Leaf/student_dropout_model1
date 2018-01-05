import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
from sklearn import model_selection
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


data = pd.read_csv('../data_feature/feature9.csv')
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
# data = data[['FACTTUITION', 'earliestchoosefrom2']]

# data = preprocessing.MinMaxScaler().fit_transform(data)


clf = AdaBoostClassifier(n_estimators=300, learning_rate=0.9)
start = time.time()
# 交叉验证
score_name = 'roc_auc'
score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y, cv=5, scoring=score_name, groups=data_y)
print(score)
print(score_name, sum(score)/len(score), 'time:', time.time()-start)




