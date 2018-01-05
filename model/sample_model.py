import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
from sklearn import model_selection
import time
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics

data = pd.read_csv('../data_feature/feature9.csv')
data_y = data['tz_students'].values

# data = data.drop(['STUDENTCODE', 'tz_students', 'weight'], axis=1)
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
score = []
for train_index, test_index in skf.split(data, data_y):
    # 训练集
    train_data = data.iloc[train_index]
    train_data = train_data[train_data.STUDYMODE_1 == 0]
    train_data['weight'] = 1
    train_data.at[train_data[train_data.tz_students == 0].index, 'weight'] = 6
    weight = train_data['weight'].values
    train_data_y = train_data['tz_students'].values
    train_data = train_data.drop(['STUDENTCODE', 'tz_students', 'weight'], axis=1)

    # 测试集
    test_data = data.iloc[test_index]
    test_y = data_y[test_index]

    # 训练模型
    clf = LGBMClassifier(num_leaves=8, learning_rate=0.05, max_depth=8, n_estimators=300, subsample=0.8,
                         colsample_bytree=1, min_child_weight=400, )
    clf.fit(X=train_data, y=train_data_y, sample_weight=weight)

    # 预测
    test_x = test_data.drop(['STUDENTCODE', 'tz_students'], axis=1)
    test_data['pre'] = 0
    test_data.at[test_data[test_data.STUDYMODE_1 == 0].index, 'pre'] = clf.predict(test_x[test_x.STUDYMODE_1 == 0])
    # tmp_score = metrics.roc_auc_score(y_true=test_y, y_score=pred)
    tmp_score = metrics.f1_score(y_true=test_y, y_pred=test_data['pre'].values, average='macro')
    # print(tmp_score)
    score.append(tmp_score)

print(score)
print('f1:', sum(score)/len(score))

