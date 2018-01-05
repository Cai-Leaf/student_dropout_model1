import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt
import time
from sklearn import model_selection


data = pd.read_csv('feature.csv')
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students', 'STUDYMODE_1', 'STUDYMODE_2'], axis=1)

# lightgbm
clf = LGBMClassifier(num_leaves=8, learning_rate=0.05, max_depth=8, n_estimators=300, subsample=0.8,
                    colsample_bytree=1, min_child_weight=1)

# 计算特征重要度
# clf.fit(X=data, y=data_y)
# score = clf.feature_importances_
# score = [(data.columns[i], score[i]) for i in range(len(score))]
# score = sorted(score, key=lambda k: k[1], reverse=True)
# for i in range(len(score)):
#     print(i, score[i])
start = time.time()
# 交叉验证
score_name = 'f1_macro'
score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y, cv=5, scoring=score_name, groups=data_y)
print(score)
print(score_name+':', sum(score)/len(score), 'time:', time.time()-start)


