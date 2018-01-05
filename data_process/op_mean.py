import pandas as pd


data = pd.read_csv('../data_origin/student_op.csv', header=0)
data = data.groupby('STUDENTCODE').mean()
data = data.drop(['tz_students'], axis=1)
data.columns = ['mean_'+name for name in data.columns]
data['STUDENTCODE'] = data.index

# data2 = pd.read_csv('../data_feature/feature.csv', header=0)
# data = pd.merge(data2, data, on='STUDENTCODE', how='left')
# print(len(data))
data.to_csv('../data_feature_lstm/feature.csv', index=False, encoding='utf-8')
