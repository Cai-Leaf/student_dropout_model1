import pandas as pd
from sklearn import model_selection
import time
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC

data = pd.read_csv('../data_feature/feature9.csv')
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
# data = preprocessing.StandardScaler().fit_transform(data)
data = preprocessing.MinMaxScaler().fit_transform(data)


# feature9 f1_macro = 0.79
clf = SVC(kernel='rbf', C=2)

start = time.time()
# 交叉验证
score_name = 'roc_auc'
score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y, cv=5, scoring=score_name, groups=data_y)
print(score)
print(score_name+':', sum(score)/len(score), 'time:', time.time()-start)


