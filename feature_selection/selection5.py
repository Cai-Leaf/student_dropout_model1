import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt
import time
from sklearn import model_selection


data = pd.read_csv('feature.csv')
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)
data = data[
    ['age', 'FACTTUITION', 'DAY_MEAN_LOGIN_NUM', 'DAY_MEAN_LOGIN_DURATION', 'earliestchoosefrom2',
     'DAY_MEAN_KJ_CLICK_NUM', 'm0_LOGIN_DURATION', 'STD_LOGIN_NUM', 'STD_LOGIN_DURATION', 'w_stdLOG_DAY',
     'rank_m1_rank_LOGIN_NUM', 'MARRIAGE_未婚', 'rank_m3_rank_LOGIN_DURATION', 'PROVINCE_NAME_广东省',
     'PROVINCE_NAME_云南省', 'rank_m5_rank_LOGIN_NUM', 'm5_LOGIN_NUM', 'rank_m0_rank_LOG_DAY', 'PROVINCE_NAME_辽宁省',
     'PROVINCE_NAME_河南省', 'PROVINCE_NAME_山西省', 'PROVINCE_NAME_湖北省', 'PROVINCE_NAME_山东省', 'STUDENTSOURCE_5',
     'PROVINCE_NAME_广西壮族自治区', 'LCENTERTYPERANK_3.0', 'LCENTERTYPERANK_4.0', 'm1_maxmin_LOG_DAY',
     'm0_maxmin_KJ_CLICK_NUM', 'm4_LM_CLICK_NUM', 'SEX_1', 'MARRIAGE_其它', 'm5_maxmin_LOGIN_DURATION',
     'm0_KJ_CLICK_NUM', 'm2_maxmin_LOG_DAY', 'LCENTERTYPERANK_2.0', 'PROVINCE_NAME_江西省', 'm5_maxmin_LOG_DAY',
     'm3_KJ_CLICK_NUM', 'm2_LOGIN_NUM', 'LCENTERTYPENAME_直属', 'm1_KJ_CLICK_NUM', 'm4_LOGIN_NUM', 'm3_LOGIN_NUM',
     'STD_LM_CLICK_NUM', 'm1_maxmin_LOGIN_DURATION', 'm4_KJ_CLICK_NUM', 'LCENTERTYPERANK_1.0', 'PROVINCE_NAME_福建省',
     'PROVINCE_NAME_内蒙古自治区', 'm0_maxmin_LOG_DAY', 'm1_LOGIN_NUM', 'm3_maxmin_LOG_DAY', 'm2_maxmin_LOGIN_NUM',
     'm0_LOGIN_NUM', 'm5_maxmin_LOGIN_NUM', 'm5_LOGIN_DURATION', 'DAY_MEAN_LM_CLICK_NUM', 'm4_maxmin_LOG_DAY',
     'm1_maxmin_LOGIN_NUM', 'mean_LOG_DAY', 'm3_maxmin_LOGIN_NUM', 'm2_maxmin_LOGIN_DURATION',
     'm3_maxmin_LOGIN_DURATION', 'm2_LOGIN_DURATION', 'm0_maxmin_LOGIN_NUM', 'w_mLOGIN_NUM', 'mean_LOGIN_NUM',
     'w_stdKJ_CLICK_NUM', 'm4_maxmin_LOGIN_NUM', 'm4_maxmin_LOGIN_DURATION', 'w_mLOG_DAY', 'm0_maxmin_LOGIN_DURATION',
     'w_mKJ_CLICK_NUM', 'mean_KJ_CLICK_NUM', 'm3_LOGIN_DURATION', 'w_stdLOGIN_NUM', 'm1_LOGIN_DURATION',
     'w_mLOGIN_DURATION', 'STD_LOG_DAY', 'm4_LOGIN_DURATION', 'mean_LOGIN_DURATION', 'w_stdLOGIN_DURATION']
]

# lightgbm
clf = LGBMClassifier(num_leaves=40, learning_rate=0.05, max_depth=20, n_estimators=300, subsample=0.8,
                     colsample_bytree=1, min_child_weight=1)

# 计算特征重要度
clf.fit(X=data, y=data_y)
score = clf.feature_importances_
score = [(data.columns[i], score[i]) for i in range(len(score))]
score = sorted(score, key=lambda k: k[1], reverse=True)
for i in range(len(score)):
    print(i, score[i])
# start = time.time()
# # 交叉验证
# score_name = 'roc_auc'
# score = model_selection.cross_val_score(estimator=clf, X=data, y=data_y,
#                                         cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=8),
#                                         scoring=score_name, groups=data_y)
# print(score)
# print(score_name+':', sum(score)/len(score), 'time:', time.time()-start)




