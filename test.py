from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import time
from lightgbm import LGBMClassifier


# 读入训练数据
data = pd.read_csv('feature_selection/feature.csv', header=0)
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
skf = StratifiedKFold(n_splits=5, random_state=6, shuffle=True)
score = []
start = time.time()
for train_index, test_index in skf.split(data, data_y):
    # 训练集
    train_data = data.iloc[train_index]
    train_y = data_y[train_index]

    # 测试集
    test_data = data.iloc[test_index]
    test_y = data_y[test_index]
    print(len(train_data), len(test_data))

    # 训练模型
    model = LGBMClassifier(num_leaves=6, learning_rate=0.05, max_depth=6, n_estimators=200, subsample=1,
                         colsample_bytree=1, min_child_weight=1)
    model.fit(X=train_data, y=train_y)

    # 预测
    pred = model.predict(test_data)
    # tmp_score = metrics.roc_auc_score(y_true=test_y, y_score=pred)
    tmp_score = metrics.f1_score(y_true=test_y, y_pred=pred, average='macro')
    print(tmp_score)
    score.append(tmp_score)

print(score)
print('f1:', sum(score)/len(score), 'time:', time.time()-start)


