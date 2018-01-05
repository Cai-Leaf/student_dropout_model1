import pandas as pd

data = pd.read_csv('../data_origin/student_msg.csv', header=0)
numerical = data[['STUDENTCODE', 'tz_students', 'age', 'FACTTUITION', 'earliestchoosefrom2']]

label = data[['STUDENTCODE', 'ENTRANCETYPE', 'LCENTERTYPENAME', 'LCENTERTYPERANK',
              'LEVELCODE', 'MARRIAGE', 'SEX', 'STUDENTSOURCE', 'STUDYMODE']]
label = label.set_index('STUDENTCODE')
numerical.at[numerical[numerical.earliestchoosefrom2 < 0].index, 'earliestchoosefrom2'] = 0
print(max(numerical['FACTTUITION'].values.tolist()))
numerical.at[numerical[numerical.FACTTUITION == 0].index, 'FACTTUITION'] = 250

for name in label.columns:
    label[name] = label[name].astype(str)

label = pd.get_dummies(label)
label['STUDENTCODE'] = label.index
data = pd.merge(numerical, label, on='STUDENTCODE', how='left')

data2 = pd.read_csv('../data_feature_lstm/feature.csv', header=0)
data = pd.merge(data, data2, on='STUDENTCODE', how='left')

print(len(data.columns))
# data.to_csv('../data_feature_lstm/feature.csv', index=False, encoding='utf-8')
