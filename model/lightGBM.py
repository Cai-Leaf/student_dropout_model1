import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
from sklearn import model_selection
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_csv('../data_feature/feature9.csv')
# data = pd.read_csv('../data_feature_test/feature3.csv')
print(len(data))
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)

# lightgbm

# feature9 auc 0.822
clf = LGBMClassifier(num_leaves=8, learning_rate=0.05, max_depth=8, n_estimators=300, subsample=0.8,
                    colsample_bytree=1, min_child_weight=2)

start = time.time()
# 交叉验证
score_name = 'roc_auc'
score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y,
                                        cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
                                        scoring=score_name, groups=data_y)
print(score)
print(score_name+':', sum(score)/len(score), 'time:', time.time()-start)

# 计算特征重要度
# clf.fit(X=data, y=data_y)
# ax = plot_importance(clf, importance_type='gain')
# plt.show()


