import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt
import time
from sklearn import model_selection

data = pd.read_csv('feature.csv')
data_stu_code = data['STUDENTCODE'].values
data_y = data['tz_students'].values
name_filter = ['FACTTUITION', 'earliestchoosefrom2', 'age', 'DAY_MEAN', 'w_m', 'STD', 'mean',
                'LCENTERTYPERANK', 'maxmin_LOGIN_NUM', 'maxmin_LOG_DAY', 'LOGIN_DURATION',
               'LCENTERTYPENAME', 'KJ_CLICK_NUM', 'STUDENTSOURCE_7', 'STUDENTSOURCE_6', 'PROVINCE_NAME_福建省',
               'PROVINCE_NAME_内蒙古自治区', 'PROVINCE_NAME_河北省', 'PROVINCE_NAME_辽宁省', 'PROVINCE_NAME_山西省',
               'PROVINCE_NAME_吉林省', 'PROVINCE_NAME_山东省', 'PROVINCE_NAME_江西省', 'w_std', 'maxmin_LOGIN_DURATION']
# name_filter = ['w_std']
name_del = ['rank', 'VIP', '普通', 'LCENTERTYPERANK_0.0', 'LCENTERTYPERANK_3.0', 'LCENTERTYPERANK_4.0',
            'STUDENTSOURCE_6', 'mean_LOG_DAY', 'mean_LM_CLICK_NUM', 'mean_KCWD_NUM', 'mean_FIN_JOB_NUM'
            'DAY_MEAN_LM_CLICK_NUM', 'DAY_MEAN_KCWD_NUM', 'STD_LM_CLICK_NUM', 'STD_KCWD_NUM', 'w_mLOGIN_NUM',
            'w_mKCWD_NUM', 'maxmin_KJ_CLICK_NUM', 'w_stdLM_CLICK_NUM', 'FIN_JOB_NUM']

names = data.columns
f_names = []
for name in names:
    for n_f in name_filter:
        if n_f in name:
            f_names.append(name)
            break

f_names2 = []
for name in f_names:
    sign = 1
    for n_f in name_del:
        if n_f in name:
            sign = 0
    if sign:
        f_names2.append(name)


for i in range(len(f_names2)):
    print(i, f_names2[i])

data = data[f_names2]
print(len(data.columns))
data['tz_students'] = data_y
data['STUDENTCODE'] = data_stu_code
print(data.info())
print(len(data.columns))
data.to_csv('feature2.csv', index=False, encoding='utf-8')

# # lightgbm
# clf = LGBMClassifier(num_leaves=6, learning_rate=0.05, max_depth=6, n_estimators=200, subsample=1,
#                     colsample_bytree=1, min_child_weight=1)
# start = time.time()
# # 交叉验证
# score_name = 'f1_macro'
# score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y, cv=5, scoring=score_name, groups=data_y)
# print(score)
# print(score_name+':', sum(score)/len(score), 'time:', time.time()-start)


