import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
from sklearn import model_selection
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_csv('../data_origin/student_grade.csv')
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)

# data = preprocessing.MinMaxScaler().fit_transform(data)


# lightgbm
# feature4 0.902
clf = LGBMClassifier(num_leaves=100, learning_rate=0.05, max_depth=15, n_estimators=400, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.1, reg_lambda=0.2)
# feature5_1: 0.9001
# clf = LGBMClassifier(num_leaves=100, learning_rate=0.05, max_depth=15, n_estimators=400, subsample=0.8,
#                     colsample_bytree=0.8, min_child_weight=1, reg_alpha=0, reg_lambda=0)
# feature5_2: 0.90000000 province_n=15
# clf = LGBMClassifier(num_leaves=39, learning_rate=0.05, max_depth=25, n_estimators=400, subsample=0.8,
#                     colsample_bytree=0.8, min_child_weight=1)
# feature5_2: 0.9003 province_n=8
# clf = LGBMClassifier(num_leaves=45, learning_rate=0.05, max_depth=15, n_estimators=400, subsample=0.8,
#                     colsample_bytree=0.8, min_child_weight=1)


start = time.time()
# # 交叉验证
# score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y, cv=5, scoring='roc_auc', groups=data_y)
# print('acc:', sum(score)/len(score), 'time:', time.time()-start)

# 计算特征重要度
clf.fit(X=data, y=data_y)
ax = plot_importance(clf)
plt.show()


