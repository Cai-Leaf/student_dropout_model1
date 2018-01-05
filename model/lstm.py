from keras.layers import Input, Dense, LSTM, Masking, Embedding, Dropout
from keras.models import Model, load_model
from keras.layers.merge import Dot, Concatenate
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import time


class LstmModel:
    def __init__(self, op_input_shape, msg_input_shape):
        op_input = Input(shape=op_input_shape, name='op_input')
        op_x = LSTM(6)(op_input)

        msg_input = Input(shape=msg_input_shape, name='msg_input')
        msg_x = Dense(6, activation='selu')(msg_input)

        x = Concatenate()([op_x, msg_x])
        x = Dense(4, activation='selu')(x)
        output = Dense(1, activation='relu', name='output')(x)

        self.model = Model(inputs=[op_input, msg_input], outputs=[output])
        self.model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])

    def fit(self, x_train_op, x_train_msg, y_train):
        self.model.fit({'op_input': np.array(x_train_op), 'msg_input': np.array(x_train_msg)},
                       {'output': np.array(y_train)},
                       # callbacks=[early_stopping, modle_check_point],
                       batch_size=100, epochs=50)

    def predict(self, test_op, test_msg):
        return self.model.predict({'op_input': np.array(test_op), 'msg_input': np.array(test_msg)})

    def save(self, filename):
        self.model.save(filename)


# 读入训练数据
data = pd.read_csv('../data_feature_lstm/feature2.csv', header=0)
stu_code1 = data['STUDENTCODE'].values
data_y = data['tz_students'].values
# data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
data = data.drop(['STUDENTCODE', 'tz_students', 'FACTTUITION', 'STUDYMODE_1', 'STUDYMODE_2'], axis=1)
# 归一化
data = preprocessing.MinMaxScaler().fit_transform(data.values)
# data = preprocessing.StandardScaler().fit_transform(data.values)

data2 = pd.read_csv('../data_origin/student_op.csv', header=0)
stu_code2 = data2['STUDENTCODE'].values
data2 = data2.drop(['tz_students', 'FIN_JOB_NUM', 'DT', 'STUDENTCODE'], axis=1)
# 归一化
data2 = preprocessing.MinMaxScaler().fit_transform(data2.values)
print(type(data2))
# data2 = preprocessing.StandardScaler().fit_transform(data2.values)

op_data = []
i = 0
while i < len(data2):
    op_data.append(data2[i:i+6])
    i += 6
op_data = np.array(op_data)

# K折交叉
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
score = []
start = time.time()
for train_index, test_index in skf.split(data, data_y):
    # 训练集
    train_op_data = op_data[train_index]
    train_msg_data = data[train_index]
    train_y = data_y[train_index]

    # 测试集
    test_op_data = op_data[test_index]
    test_msg_data = data[test_index]
    test_y = data_y[test_index]

    # 训练模型
    model = LstmModel(op_input_shape=train_op_data[0].shape,
                      msg_input_shape=train_msg_data[0].shape)
    model.fit(x_train_op=train_op_data,
              x_train_msg=train_msg_data,
              y_train=train_y)

    # 预测
    pred = model.predict(test_op_data, test_msg_data)
    pred = np.array([i[0] for i in pred])
    pred = np.where(pred > 0.55, pred, 0)
    pred = np.where(pred <= 0.55, pred, 1)
    tmp_score = metrics.roc_auc_score(y_true=test_y, y_score=pred)
    # tmp_score = metrics.f1_score(y_true=test_y, y_pred=pred, average='macro')
    print(tmp_score)
    score.append(tmp_score)

print(score)
print('f1:', sum(score)/len(score), 'time:', time.time()-start)
