import pandas as pd
import numpy as np


data = pd.read_csv('../data_origin/student_op.csv', header=0)
data = data.set_index('STUDENTCODE')
data = data.drop(['tz_students', 'FIN_JOB_NUM', 'DT'], axis=1)

w_mean_data = []
index_name = []
i = 0
while i < len(data):
    w_mean_data.append(np.mean(data.iloc[i:i+6].values * np.array([[1], [1], [1], [2], [2], [6]]), axis=0))
    index_name.append(data.index[i])
    i += 6
data = pd.DataFrame(w_mean_data, index_name, columns=['w_m'+name for name in data.columns])
data['STUDENTCODE'] = data.index


data2 = pd.read_csv('../data_feature/feature9.csv', header=0)
data = pd.merge(data2, data, on='STUDENTCODE', how='left')
print(len(data))
data.to_csv('../feature_selection/feature.csv', index=False, encoding='utf-8')
