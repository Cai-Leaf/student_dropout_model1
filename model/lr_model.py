import pandas as pd
from sklearn import model_selection
import time
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('../data_feature/feature9.csv')
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
data = preprocessing.StandardScaler().fit_transform(data)
# data = preprocessing.MinMaxScaler().fit_transform(data)


# feature9 auc0.826
clf = LogisticRegression(C=2, intercept_scaling=1)

start = time.time()
# 交叉验证
score_name = 'roc_auc'
score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y,
                                        cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
                                        scoring=score_name)
print(score)
print(score_name+':', sum(score)/len(score), 'time:', time.time()-start)


