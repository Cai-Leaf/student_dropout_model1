import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import time
from sklearn import model_selection


data = pd.read_csv('feature.csv', header=0)
# data = pd.read_csv('../data_feature_test/feature4.csv', header=0)
# data = pd.read_csv('../data_feature/feature7_1.csv', header=0)
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
# data.at[data[data.FACTTUITION ==250].index, 'FACTTUITION'] = 0

# drop_list = ['w_stdKCWD_NUM', 'm0_maxmin_LOG_DAY', 'm1_maxmin_LOG_DAY', 'm2_maxmin_LOG_DAY', 'm3_maxmin_LOG_DAY',
#              'm4_maxmin_LOG_DAY', 'm5_maxmin_LOG_DAY', 'DAY_MEAN_LM_CLICK_NUM']
# data = data.drop(drop_list, axis=1)
# print(data.info())
# print(len(data.columns))
# data.to_csv('feature_final.csv', index=False, encoding='utf-8')

# lightgbm
# clf = LGBMClassifier(num_leaves=15, learning_rate=0.05, max_depth=15, n_estimators=200, subsample=1,
#                     colsample_bytree=1, min_child_weight=1)

# xgboost
clf = XGBClassifier(learning_rate=0.05, max_depth=50,  n_estimators=300, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=1, gamma=0)


start = time.time()
# 交叉验证
# model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=2333)lee
score_name = 'f1'
score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y,
                                        cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
                                        scoring=score_name)
print(score)
print(score_name+':', sum(score)/len(score), 'time:', time.time()-start)

# 计算特征重要度
# clf.fit(X=data, y=data_y)
# score = clf.feature_importances_
# score = [(data.columns[i], score[i]) for i in range(len(score))]
# score = sorted(score, key=lambda k: k[1], reverse=True)
# for i in range(len(score)):
#     print(i, score[i])


