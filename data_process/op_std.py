import pandas as pd


data = pd.read_csv('../data_origin/student_op.csv', header=0)
data = data.drop(['FIN_JOB_NUM', 'tz_students'], axis=1)
data = data.groupby('STUDENTCODE').std()
data.columns = ['STD_'+name for name in data.columns]
data['STUDENTCODE'] = data.index

print(data.info())

# data2 = pd.read_csv('../data_feature/feature9.csv', header=0)
# data = pd.merge(data2, data, on='STUDENTCODE', how='left')
# print(data.columns)
# data.to_csv('../data_feature/feature9.csv', index=False, encoding='utf-8')


