import pandas as pd


data = pd.read_csv('../data_origin/student_op.csv', header=0)

month_data0 = data[data.DT == '2016-11-01']
month_data1 = data[data.DT == '2016-12-01']
month_data2 = data[data.DT == '2017-01-01']
month_data3 = data[data.DT == '2017-02-01']
month_data4 = data[data.DT == '2017-03-01']
month_data5 = data[data.DT == '2017-04-01']
del data

month_data = [month_data0, month_data1, month_data2, month_data3, month_data4, month_data5]
for i in range(len(month_data)):
    month_data[i] = month_data[i].drop(['DT', 'tz_students'], axis=1)
for i in range(len(month_data)):
    tmp = month_data[i].columns
    tmp = ['STUDENTCODE']+['m'+str(i)+'_'+col for col in tmp[1:]]
    month_data[i].columns = tmp

data = month_data[0]
for i in range(1, len(month_data)):
    data = pd.merge(data, month_data[i], on='STUDENTCODE', how='left')
data.to_csv('../data_feature/feature.csv', index=False, encoding='utf-8')
