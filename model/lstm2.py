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
    def __init__(self, msg_input_shape):
        login_num_input = Input(shape=(6, 1), name='login_num_input')
        login_num_x = LSTM(1)(login_num_input)

        log_day_input = Input(shape=(6, 1), name='log_day_input')
        log_day_x = LSTM(1)(log_day_input)

        log_duration_input = Input(shape=(6, 1), name='log_duration_input')
        log_duration_x = LSTM(1)(log_duration_input)

        lmclick_num_input = Input(shape=(6, 1), name='lmclick_num_input')
        lmclick_num_x = LSTM(1)(lmclick_num_input)

        kjclick_num_input = Input(shape=(6, 1), name='kjclick_num_input')
        kjclick_num_x = LSTM(1)(kjclick_num_input)

        kcwd_num_input = Input(shape=(6, 1), name='kcwd_num_input')
        kcwd_num_x = LSTM(1)(kcwd_num_input)

        msg_input = Input(shape=msg_input_shape, name='msg_input')
        msg_x = Dense(6, activation='selu')(msg_input)

        x = Concatenate()([login_num_x, log_day_x, log_duration_x, lmclick_num_x, kjclick_num_x, kcwd_num_x, msg_x])
        x = Dense(4, activation='selu')(x)
        output = Dense(1, activation='sigmoid', name='output')(x)

        self.model = Model(inputs=[login_num_input, log_day_input, log_duration_input,
                                   lmclick_num_input, kjclick_num_input, kcwd_num_input, msg_input],
                           outputs=[output])
        self.model.compile(optimizer='rmsprop',
                      loss='mse',
                      metrics=['acc'])

    def fit(self, login_num, log_day, log_duration, lmclick_num, kjclick_num, kcwd_num, msg, y_train):
        self.model.fit({'login_num_input': np.array(login_num),
                        'log_day_input': np.array(log_day),
                        'log_duration_input': np.array(log_duration),
                        'lmclick_num_input': np.array(lmclick_num),
                        'kjclick_num_input': np.array(kjclick_num),
                        'kcwd_num_input': np.array(kcwd_num),
                        'msg_input': np.array(msg)},
                       {'output': np.array(y_train)},
                       # callbacks=[early_stopping, modle_check_point],
                       batch_size=100, epochs=100)

    def predict(self, login_num, log_day, log_duration, lmclick_num, kjclick_num, kcwd_num, msg):
        return self.model.predict({'login_num_input': np.array(login_num),
                                   'log_day_input': np.array(log_day),
                                   'log_duration_input': np.array(log_duration),
                                   'lmclick_num_input': np.array(lmclick_num),
                                   'kjclick_num_input': np.array(kjclick_num),
                                   'kcwd_num_input': np.array(kcwd_num),
                                   'msg_input': np.array(msg)},)

    def save(self, filename):
        self.model.save(filename)


# 读入训练数据
data = pd.read_csv('../data_feature_lstm/feature2.csv', header=0)
stu_code1 = data['STUDENTCODE'].values
data_y = data['tz_students'].values
# data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
data = data.drop(['STUDENTCODE', 'tz_students', 'FACTTUITION', 'STUDYMODE_2'], axis=1)
# 归一化
# data = preprocessing.MinMaxScaler().fit_transform(data.values)
data = preprocessing.StandardScaler().fit_transform(data.values)

# data2 = pd.read_csv('../data_feature_test/student_op_rank.csv', header=0)
# stu_code2 = data2['STUDENTCODE'].values
# data2 = data2.drop(['DT', 'STUDENTCODE'], axis=1)
# data2.columns = [name[5:] for name in data2.columns]
# print(data2.info())

data2 = pd.read_csv('../data_origin/student_op.csv', header=0)
stu_code2 = data2['STUDENTCODE'].values
data2 = data2.drop(['tz_students', 'FIN_JOB_NUM', 'DT', 'STUDENTCODE'], axis=1)
data2[data2.columns] = preprocessing.StandardScaler().fit_transform(data2.values)
print(data2.info())

login_num_data = []
log_day_data = []
log_duration_data = []
lmclick_num_data = []
kjclick_num_data = []
kcwd_num_data = []
i = 0
while i < len(data2):
    login_num_data.append(data2[i:i+6][['LOGIN_NUM']].values)
    log_day_data.append(data2[i:i + 6][['LOG_DAY']].values)
    log_duration_data.append(data2[i:i + 6][['LOGIN_DURATION']].values)
    lmclick_num_data.append(data2[i:i + 6][['LM_CLICK_NUM']].values)
    kjclick_num_data.append(data2[i:i + 6][['KJ_CLICK_NUM']].values)
    kcwd_num_data.append(data2[i:i + 6][['KCWD_NUM']].values)
    i += 6

login_num_data = np.array(login_num_data)
log_day_data = np.array(log_day_data)
log_duration_data = np.array(log_duration_data)
lmclick_num_data = np.array(lmclick_num_data)
kjclick_num_data = np.array(kjclick_num_data)
kcwd_num_data = np.array(kcwd_num_data)


# K折交叉
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
score = []
start = time.time()
for train_index, test_index in skf.split(data, data_y):
    # 训练集
    train_msg_data = data[train_index]
    train_y = data_y[train_index]
    train_login_num_data = login_num_data[train_index]
    train_log_day_data = log_day_data[train_index]
    train_log_duration_data = log_duration_data[train_index]
    train_lmclick_num_data = lmclick_num_data[train_index]
    train_kjclick_num_data = kjclick_num_data[train_index]
    train_kcwd_num_data = kcwd_num_data[train_index]

    # 测试集
    test_msg_data = data[test_index]
    test_y = data_y[test_index]
    test_login_num_data = login_num_data[test_index]
    test_log_day_data = log_day_data[test_index]
    test_log_duration_data = log_duration_data[test_index]
    test_lmclick_num_data = lmclick_num_data[test_index]
    test_kjclick_num_data = kjclick_num_data[test_index]
    test_kcwd_num_data = kcwd_num_data[test_index]

    # 训练模型
    model = LstmModel(msg_input_shape=train_msg_data[0].shape)
    model.fit(login_num=train_login_num_data,
              log_day=train_log_day_data,
              log_duration=train_log_duration_data,
              lmclick_num=train_lmclick_num_data,
              kjclick_num=train_kjclick_num_data,
              kcwd_num=train_kcwd_num_data,
              msg=train_msg_data,
              y_train=train_y)

    # 预测
    pred = model.predict(login_num=test_login_num_data,
                         log_day=test_log_day_data,
                         log_duration=test_log_duration_data,
                         lmclick_num=test_lmclick_num_data,
                         kjclick_num=test_kjclick_num_data,
                         kcwd_num=test_kcwd_num_data,
                         msg=test_msg_data,)
    pred = np.array([i[0] for i in pred])
    pred = np.where(pred > 0.5, pred, 0)
    pred = np.where(pred <= 0.5, pred, 1)
    # tmp_score = metrics.roc_auc_score(y_true=test_y, y_score=pred)
    tmp_score = metrics.f1_score(y_true=test_y, y_pred=pred, average='macro')
    print(tmp_score)
    score.append(tmp_score)

print(score)
print('f1:', sum(score)/len(score), 'time:', time.time()-start)
