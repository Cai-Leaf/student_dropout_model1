import pandas as pd

data = pd.read_csv('../data_origin/student_grade.csv', header=0)
# data2 = pd.read_csv('../data_feature/feature2.csv', header=0)

data = data.drop(['tz_students'], axis=1)
# data = pd.merge(data, data2, on='STUDENTCODE', how='left')
# print(len(data))
# data.to_csv('../data_feature/feature3.csv', index=False, encoding='utf-8')