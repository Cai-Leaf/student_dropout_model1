import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
from sklearn import model_selection
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('../data_feature/feature7_1.csv')
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
# data = data[['FACTTUITION', 'earliestchoosefrom2']]

# data = preprocessing.MinMaxScaler().fit_transform(data)


clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=3, random_state=0)

start = time.time()
# 交叉验证
score_name = 'f1_macro'
score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y, cv=5, scoring=score_name, groups=data_y)
print(score)
print(score_name, sum(score)/len(score), 'time:', time.time()-start)




