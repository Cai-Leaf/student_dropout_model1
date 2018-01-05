import pandas as pd
import numpy as np


data = pd.read_csv('../data_origin/student_op.csv', header=0)
data = data.drop(['tz_students', 'FIN_JOB_NUM'], axis=1)

# month_data0 = data[data.DT == '2016-11-01']
# month_data1 = data[data.DT == '2016-12-01']
# month_data2 = data[data.DT == '2017-01-01']
# month_data3 = data[data.DT == '2017-02-01']
# month_data4 = data[data.DT == '2017-03-01']
# month_data5 = data[data.DT == '2017-04-01']
#
# month_data = [month_data0, month_data1, month_data2, month_data3, month_data4, month_data5]
# for i in range(len(month_data)):
#     month_data[i] = month_data[i].drop(['DT', 'STUDENTCODE'], axis=1)
#
# op_data = []
# for i in range(len(month_data[0])):
#     tmp = np.zeros((6, 6))
#     for j in range(len(month_data)):
#         tmp[j] = month_data[j].iloc[i].values
#     op_data.append(tmp)
#
# print(len(op_data))

# data = month_data[0]
# for i in range(1, len(month_data)):
#     data = pd.merge(data, month_data[i], on='STUDENTCODE', how='left')
# data.to_csv('../data_feature/feature.csv', index=False, encoding='utf-8')
op_data2 = []
data = data.drop(['DT', 'STUDENTCODE'], axis=1)
i = 0
while i < len(data):
    op_data2.append(data.iloc[i:i+6].values)
    i += 6

