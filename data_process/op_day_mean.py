import pandas as pd


data = pd.read_csv('../data_origin/student_op.csv', header=0)
data = data.drop(['FIN_JOB_NUM', 'tz_students'], axis=1)
data = data.groupby('STUDENTCODE').sum()

data['LOGIN_NUM'] = data['LOGIN_NUM'].values/(data['LOG_DAY'].values + 0.00000001)
data['LOGIN_DURATION'] = data['LOGIN_DURATION'].values/(data['LOG_DAY'].values + 0.00000001)
data['LM_CLICK_NUM'] = data['LM_CLICK_NUM'].values/(data['LOG_DAY'].values + 0.00000001)
data['KJ_CLICK_NUM'] = data['KJ_CLICK_NUM'].values/(data['LOG_DAY'].values + 0.00000001)
data['KCWD_NUM'] = data['KCWD_NUM'].values/(data['LOG_DAY'].values + 0.00000001)
data = data.drop(['LOG_DAY'], axis=1)
data.columns = ['DAY_MEAN_'+name for name in data.columns]

data['STUDENTCODE'] = data.index
data2 = pd.read_csv('../data_feature/feature7_1.csv', header=0)
data = pd.merge(data2, data, on='STUDENTCODE', how='left')
print(data.columns)
data.to_csv('../data_feature/feature9.csv', index=False, encoding='utf-8')
