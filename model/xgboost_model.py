import pandas as pd
from sklearn import model_selection
import time
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('../feature_selection/feature.csv')
# data = pd.read_csv('../data_feature/feature9.csv')
# data = pd.read_csv('../data_feature_test/feature3.csv')
print(len(data))
print(len(data.columns))
data_y = data['tz_students'].values
data_y = 1-data_y
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
# data = data.drop(['w_mLOGIN_NUM', 'w_mLOG_DAY', 'w_mLOGIN_DURATION', 'w_mLM_CLICK_NUM',
#                   'w_mKJ_CLICK_NUM', 'w_mKCWD_NUM'], axis=1)
# data = data.drop(['STUDENTCODE', 'tz_students', 'FACTTUITION', 'STUDYMODE_1', 'STUDYMODE_2'], axis=1)


# feature7_1 auc0.826
clf = XGBClassifier(learning_rate=0.05, max_depth=30,  n_estimators=300, subsample=0.8,
                    colsample_bytree=1, min_child_weight=1, gamma=0)


start = time.time()
# 交叉验证
score_name = 'f1'
score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y,
                                        cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
                                        scoring=score_name, groups=data_y)
print(score)
print(score_name+':', sum(score)/len(score), 'time:', time.time()-start)

# 计算特征重要度
# clf.fit(X=data, y=data_y)
# ax = plot_importance(clf, importance_type='gain')
# plt.show()
##########################################################################################################
# x_train, x_test, y_train, y_test = train_test_split(data, data_y, test_size=0.2, random_state=1)
# dtrain = xgb.DMatrix(data=x_train, label=y_train)
# dtest = xgb.DMatrix(data=x_test, label=y_test)
# param = {'max_depth': 6,
#          'eta': 0.05,
#          'silent': 1,
#          'min_child_weight ': 1,
#          'objective': 'binary:logistic',
#          'subsample ': 0.8,
#          'colsample_bytree ': 0.8,
#          }
# num_round = 300
# evallist = [(dtest, 'eval'), (dtrain, 'train')]
# bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=10)
#
# preds = bst.predict(dtest)
# tmp_score = metrics.roc_auc_score(y_true=y_test, y_score=preds)
# print(tmp_score)

