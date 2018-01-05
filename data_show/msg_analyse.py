import pandas as pd

data = pd.read_csv('../data_origin/student_msg.csv', header=0)
# data = data[data.STUDYMODE == 2]
print(len(data[(data.tz_students == 0) & (data.STUDYMODE == 1)]))
print(len(data[(data.STUDYMODE == 1)]))
# print(len(data))
# print(len(data[(data.tz_students == 1)]))
print(len(set(data['PROVINCE_NAME'].values.tolist())))
