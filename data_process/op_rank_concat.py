import pandas as pd
import numpy as np


def get_rank(val):
    tmp_dic = {}
    j = len(val)
    for index in sorted(val, reverse=True):
        tmp_dic[index] = j
        j -= 1
    return np.array([tmp_dic[index] for index in val])

# data = pd.read_csv('../data_origin/student_op.csv', header=0)
# data = data.set_index('STUDENTCODE')
# dt = data['DT'].values
# data = data.drop(['tz_students', 'FIN_JOB_NUM', 'DT'], axis=1)
# new_val = np.zeros((len(data), 6))
# i = 0
# while i < len(data):
#     tmp_data = []
#     for name in data.columns:
#         tmp_data.append(get_rank(data.iloc[i:i+6][name].values))
#     tmp_data = np.array(tmp_data).T
#     new_val[i:i+6] = tmp_data
#     i += 6
#
# data = pd.DataFrame(new_val, index=data.index, columns=['rank_'+name for name in data.columns])
# data['DT'] = dt
# data.to_csv('../data_feature_test/student_op_rank.csv', encoding='utf-8')
# print(data.iloc[6:12])

data = pd.read_csv('../data_feature_test/student_op_rank.csv', header=0)

month_data0 = data[data.DT == '2016-11-01']
month_data1 = data[data.DT == '2016-12-01']
month_data2 = data[data.DT == '2017-01-01']
month_data3 = data[data.DT == '2017-02-01']
month_data4 = data[data.DT == '2017-03-01']
month_data5 = data[data.DT == '2017-04-01']
del data

month_data = [month_data0, month_data1, month_data2, month_data3, month_data4, month_data5]
for i in range(len(month_data)):
    month_data[i] = month_data[i].drop(['DT'], axis=1)
for i in range(len(month_data)):
    tmp = month_data[i].columns
    tmp = ['STUDENTCODE']+['rank_m'+str(i)+'_'+col for col in tmp[1:]]
    month_data[i].columns = tmp

data = month_data[0]
for i in range(1, len(month_data)):
    data = pd.merge(data, month_data[i], on='STUDENTCODE', how='left')

data2 = pd.read_csv('../feature_selection/feature.csv', header=0)
data = pd.merge(data2, data, on='STUDENTCODE', how='left')
print(len(data))
data.to_csv('../feature_selection/feature.csv', index=False, encoding='utf-8')
