import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt
import time
from sklearn import model_selection
from xgboost import XGBClassifier


data = pd.read_csv('feature.csv')
data_y = data['tz_students'].values
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)

# lightgbm
# clf = LGBMClassifier(num_leaves=20, learning_rate=0.05, max_depth=15, n_estimators=300, subsample=0.8,
#                     colsample_bytree=1, min_child_weight=1)
# 0.873336146325
clf = XGBClassifier(learning_rate=0.05, max_depth=20,  n_estimators=300, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=1, gamma=0)


# 计算特征重要度
clf.fit(X=data, y=data_y)
score = clf.feature_importances_
score = [(data.columns[i], score[i]) for i in range(len(score))]
score = sorted(score, key=lambda k: k[1], reverse=True)
name_list = []
for i in range(len(score)):
    if score[i][1] > 0:
        name_list.append(score[i][0])
        print(i, score[i])
    else:
        break
print(name_list)
#
# feature = name_list[0:10]
# cur_score = 0
# # 特征选择
# for i in range(10, len(name_list)):
#     save_feature = name_list[i]
#     tmp_list = feature+[save_feature]
#     tmp_data = data[tmp_list]
#     # 交叉验证
#     score_name = 'roc_auc'
#     # tmp_clf = LGBMClassifier(num_leaves=30, learning_rate=0.05, max_depth=20, n_estimators=300, subsample=0.8,
#     #                          colsample_bytree=1, min_child_weight=1)
#     tmp_clf = XGBClassifier(learning_rate=0.05, max_depth=20, n_estimators=300, subsample=0.8,
#                             colsample_bytree=0.8, min_child_weight=1, gamma=0)
#     score = model_selection.cross_val_score(estimator=tmp_clf, X=tmp_data, y=data_y,
#                                             cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
#                                             scoring=score_name, groups=data_y)
#     mean_score = sum(score)/len(score)
#     print(i, score_name + ':', mean_score)
#     if mean_score > cur_score:
#         feature.append(save_feature)
#         cur_score = mean_score
#     else:
#         print('del', save_feature)
#     print('feature_num:', len(feature), 'cur score:', cur_score)
#     print('_________________________________________________________________________________________________')
#
# print(name_list)
# ['age', 'FACTTUITION', 'DAY_MEAN_LOGIN_NUM', 'DAY_MEAN_LOGIN_DURATION', 'earliestchoosefrom2', 'DAY_MEAN_KJ_CLICK_NUM',
#  'm0_LOGIN_DURATION', 'STD_LOGIN_NUM', 'STD_LOGIN_DURATION', 'w_stdLOG_DAY', 'w_stdLOGIN_DURATION',
#  'mean_LOGIN_DURATION', 'm4_LOGIN_DURATION', 'STD_LOG_DAY', 'w_mLOGIN_DURATION', 'm1_LOGIN_DURATION', 'w_stdLOGIN_NUM',
#  'm3_LOGIN_DURATION', 'STD_KJ_CLICK_NUM', 'mean_KJ_CLICK_NUM', 'w_mKJ_CLICK_NUM', 'm0_maxmin_LOGIN_DURATION',
#  'w_mLOG_DAY', 'm4_maxmin_LOGIN_DURATION', 'm4_maxmin_LOGIN_NUM', 'w_stdKJ_CLICK_NUM', 'mean_LOGIN_NUM',
#  'w_mLOGIN_NUM', 'm0_maxmin_LOGIN_NUM', 'm2_LOGIN_DURATION', 'm3_maxmin_LOGIN_DURATION', 'm2_maxmin_LOGIN_DURATION',
#  'm3_maxmin_LOGIN_NUM', 'mean_LOG_DAY', 'm1_maxmin_LOGIN_NUM', 'm4_maxmin_LOG_DAY', 'DAY_MEAN_LM_CLICK_NUM',
#  'm5_LOGIN_DURATION', 'm5_maxmin_LOGIN_NUM', 'm0_LOGIN_NUM', 'm2_maxmin_LOGIN_NUM', 'm3_maxmin_LOG_DAY', 'm1_LOGIN_NUM',
#  'm0_maxmin_LOG_DAY', 'PROVINCE_NAME_内蒙古自治区', 'PROVINCE_NAME_福建省', 'LCENTERTYPERANK_1.0', 'm4_KJ_CLICK_NUM',
#  'm1_maxmin_LOGIN_DURATION', 'STD_LM_CLICK_NUM', 'm3_LOGIN_NUM', 'm4_LOGIN_NUM', 'm1_KJ_CLICK_NUM',
#  'LCENTERTYPENAME_直属', 'm2_LOGIN_NUM', 'm3_KJ_CLICK_NUM', 'm5_maxmin_LOG_DAY', 'PROVINCE_NAME_江西省',
#  'LCENTERTYPERANK_2.0', 'm2_maxmin_LOG_DAY', 'm0_KJ_CLICK_NUM', 'm5_maxmin_LOGIN_DURATION', 'MARRIAGE_其它',
#  'SEX_1', 'm4_LM_CLICK_NUM', 'm0_maxmin_KJ_CLICK_NUM', 'm1_maxmin_LOG_DAY', 'PROVINCE_NAME_河北省',
#  'LCENTERTYPERANK_4.0', 'LCENTERTYPERANK_3.0', 'm1_LOG_DAY', 'PROVINCE_NAME_广西壮族自治区', 'STUDENTSOURCE_5',
#  'PROVINCE_NAME_山东省', 'PROVINCE_NAME_湖北省', 'STUDENTSOURCE_6', 'PROVINCE_NAME_山西省', 'm4_maxmin_LM_CLICK_NUM',
#  'PROVINCE_NAME_河南省', 'm2_KJ_CLICK_NUM', 'w_stdLM_CLICK_NUM', 'm4_maxmin_KJ_CLICK_NUM', 'MARRIAGE_已婚',
#  'm0_LOG_DAY', 'PROVINCE_NAME_辽宁省', 'rank_m0_rank_LOG_DAY', 'LCENTERTYPENAME_普通', 'm5_LOGIN_NUM',
#  'PROVINCE_NAME_吉林省', 'rank_m3_rank_LOG_DAY', 'PROVINCE_NAME_湖南省', 'LEVELCODE_1', 'rank_m5_rank_LOGIN_NUM',
#  'mean_LM_CLICK_NUM', 'PROVINCE_NAME_江苏省', 'm3_maxmin_LM_CLICK_NUM', 'm1_LM_CLICK_NUM', 'm3_LM_CLICK_NUM',
#  'w_mLM_CLICK_NUM', 'rank_m0_rank_LOGIN_NUM', 'm4_LOG_DAY', 'm5_maxmin_LM_CLICK_NUM', 'rank_m1_rank_LOG_DAY',
#  'rank_m4_rank_LOGIN_NUM', 'rank_m4_rank_LOG_DAY', 'm3_LOG_DAY', 'PROVINCE_NAME_云南省', 'PROVINCE_NAME_广东省',
#  'm1_maxmin_KJ_CLICK_NUM', 'm2_maxmin_LM_CLICK_NUM', 'm3_maxmin_KJ_CLICK_NUM', 'm2_LM_CLICK_NUM', 'STUDENTSOURCE_7',
#  'm1_maxmin_LM_CLICK_NUM', 'rank_m2_rank_LOG_DAY', 'rank_m3_rank_LOGIN_NUM', 'rank_m2_rank_LOGIN_NUM', 'm0_LM_CLICK_NUM',
#  'm5_maxmin_KJ_CLICK_NUM', 'm0_maxmin_LM_CLICK_NUM', 'LEVELCODE_3', 'PROVINCE_NAME_北京市', 'rank_m4_rank_LOGIN_DURATION',
#  'PROVINCE_NAME_贵州省', 'm2_maxmin_KJ_CLICK_NUM', 'PROVINCE_NAME_新疆维吾尔自治区', 'rank_m0_rank_LOGIN_DURATION',
#  'rank_m3_rank_LOGIN_DURATION', 'rank_m5_rank_LOGIN_DURATION', 'MARRIAGE_未婚', 'SEX_2', 'm2_LOG_DAY', 'm5_LM_CLICK_NUM',
#  'PROVINCE_NAME_安徽省', 'rank_m1_rank_LOGIN_NUM', 'm5_LOG_DAY', 'rank_m0_rank_KJ_CLICK_NUM', 'rank_m1_rank_LOGIN_DURATION',
#  'ENTRANCETYPE_1', 'm5_KJ_CLICK_NUM', 'PROVINCE_NAME_浙江省', 'rank_m2_rank_LOGIN_DURATION', 'rank_m5_rank_LOG_DAY',
#  'LCENTERTYPERANK_0.0', 'PROVINCE_NAME_宁夏回族自治区', 'MARRIAGE_其他', 'PROVINCE_NAME_甘肃省',
#  'rank_m0_rank_LM_CLICK_NUM', 'PROVINCE_NAME_黑龙江省', 'rank_m1_rank_KJ_CLICK_NUM', 'PROVINCE_NAME_天津市',
#  'rank_m2_rank_KJ_CLICK_NUM', 'rank_m3_rank_KJ_CLICK_NUM', 'rank_m1_rank_LM_CLICK_NUM', 'STUDYMODE_2',
#  'rank_m3_rank_LM_CLICK_NUM', 'rank_m4_rank_LM_CLICK_NUM', 'LCENTERTYPENAME_VIP', 'STUDENTSOURCE_10',
#  'PROVINCE_NAME_重庆市', 'rank_m4_rank_KJ_CLICK_NUM', 'rank_m5_rank_LM_CLICK_NUM', 'PROVINCE_NAME_青海省',
#  'rank_m2_rank_LM_CLICK_NUM', 'PROVINCE_NAME_陕西省', 'rank_m5_rank_KJ_CLICK_NUM', 'PROVINCE_NAME_四川省',
#  'm0_FIN_JOB_NUM', 'm1_FIN_JOB_NUM', 'm4_FIN_JOB_NUM', 'mean_FIN_JOB_NUM']

# ['FACTTUITION', 'age', 'earliestchoosefrom2', 'DAY_MEAN_LOGIN_NUM', 'w_stdLOG_DAY', 'DAY_MEAN_LOGIN_DURATION', 'DAY_MEAN_KJ_CLICK_NUM', 'm1_LOGIN_DURATION', 'm4_LOGIN_DURATION', 'STD_LOGIN_NUM', 'm3_LOGIN_DURATION', 'm0_LOGIN_DURATION', 'mean_KJ_CLICK_NUM', 'STD_LOG_DAY', 'w_mKJ_CLICK_NUM', 'w_stdLOGIN_NUM', 'm0_maxmin_LOGIN_DURATION', 'mean_LOGIN_DURATION', 'w_stdLOGIN_DURATION', 'w_mLOGIN_DURATION', 'STD_LOGIN_DURATION', 'm4_maxmin_LOGIN_NUM', 'STD_KJ_CLICK_NUM', 'PROVINCE_NAME_福建省', 'w_mLOG_DAY', 'mean_LOGIN_NUM', 'PROVINCE_NAME_内蒙古自治区', 'm3_maxmin_LOGIN_DURATION', 'm4_maxmin_LOGIN_DURATION', 'w_mLOGIN_NUM', 'w_stdKJ_CLICK_NUM', 'm2_LOGIN_DURATION', 'm5_LOGIN_DURATION', 'm5_maxmin_LOGIN_NUM', 'm3_maxmin_LOGIN_NUM', 'm3_maxmin_LOG_DAY', 'm1_maxmin_LOGIN_NUM', 'm2_maxmin_LOGIN_DURATION', 'm4_maxmin_LOG_DAY', 'LCENTERTYPENAME_直属', 'LCENTERTYPERANK_1.0', 'm0_KJ_CLICK_NUM', 'm4_LOGIN_NUM', 'm4_KJ_CLICK_NUM', 'PROVINCE_NAME_河北省', 'm1_KJ_CLICK_NUM', 'm2_maxmin_LOG_DAY', 'm0_maxmin_LOG_DAY', 'm0_maxmin_KJ_CLICK_NUM', 'm0_maxmin_LOGIN_NUM', 'STD_LM_CLICK_NUM', 'm1_maxmin_LOGIN_DURATION', 'LCENTERTYPERANK_2.0', 'm0_LOGIN_NUM', 'm1_LOGIN_NUM', 'mean_LOG_DAY', 'LCENTERTYPERANK_4.0', 'STUDENTSOURCE_5', 'm2_KJ_CLICK_NUM', 'PROVINCE_NAME_江西省', 'm4_maxmin_LM_CLICK_NUM', 'm3_KJ_CLICK_NUM', 'PROVINCE_NAME_山东省', 'm3_LOGIN_NUM', 'm5_LOGIN_NUM', 'm2_maxmin_LOGIN_NUM', 'm5_maxmin_LOG_DAY', 'MARRIAGE_已婚', 'DAY_MEAN_LM_CLICK_NUM', 'PROVINCE_NAME_辽宁省', 'PROVINCE_NAME_山西省', 'm5_maxmin_LOGIN_DURATION', 'PROVINCE_NAME_吉林省', 'PROVINCE_NAME_广西壮族自治区', 'PROVINCE_NAME_湖北省', 'PROVINCE_NAME_湖南省', 'MARRIAGE_其它', 'm1_LOG_DAY', 'm2_maxmin_LM_CLICK_NUM', 'm4_maxmin_KJ_CLICK_NUM', 'STUDENTSOURCE_6', 'STUDENTSOURCE_7', 'rank_m0_rank_LOG_DAY', 'LCENTERTYPENAME_普通', 'LCENTERTYPERANK_3.0', 'PROVINCE_NAME_河南省', 'm2_LOGIN_NUM', 'PROVINCE_NAME_云南省', 'm3_LOG_DAY', 'mean_LM_CLICK_NUM', 'm0_LOG_DAY', 'm1_LM_CLICK_NUM', 'm4_LOG_DAY', 'm4_LM_CLICK_NUM', 'PROVINCE_NAME_江苏省', 'm1_maxmin_LOG_DAY', 'rank_m3_rank_LOG_DAY', 'm1_maxmin_KJ_CLICK_NUM', 'm5_maxmin_KJ_CLICK_NUM', 'rank_m1_rank_LOG_DAY', 'PROVINCE_NAME_广东省', 'w_stdLM_CLICK_NUM', 'rank_m4_rank_LOGIN_NUM', 'm3_LM_CLICK_NUM', 'PROVINCE_NAME_北京市', 'PROVINCE_NAME_新疆维吾尔自治区', 'w_mLM_CLICK_NUM', 'LEVELCODE_1', 'PROVINCE_NAME_贵州省', 'rank_m4_rank_LOG_DAY', 'm0_maxmin_LM_CLICK_NUM', 'm3_maxmin_LM_CLICK_NUM', 'rank_m2_rank_LOGIN_NUM', 'SEX_1', 'm0_LM_CLICK_NUM', 'm2_LM_CLICK_NUM', 'PROVINCE_NAME_浙江省', 'rank_m2_rank_LOG_DAY', 'rank_m4_rank_LOGIN_DURATION', 'm5_maxmin_LM_CLICK_NUM', 'rank_m3_rank_LOGIN_NUM', 'rank_m5_rank_LOGIN_DURATION', 'MARRIAGE_未婚', 'm1_maxmin_LM_CLICK_NUM', 'rank_m0_rank_LOGIN_NUM', 'rank_m0_rank_KJ_CLICK_NUM', 'rank_m5_rank_LOGIN_NUM', 'ENTRANCETYPE_1', 'm5_LOG_DAY', 'PROVINCE_NAME_黑龙江省', 'rank_m0_rank_LOGIN_DURATION', 'rank_m3_rank_LOGIN_DURATION', 'm5_LM_CLICK_NUM', 'm5_KJ_CLICK_NUM', 'PROVINCE_NAME_安徽省', 'LEVELCODE_3', 'm2_maxmin_KJ_CLICK_NUM', 'm3_maxmin_KJ_CLICK_NUM', 'rank_m1_rank_LOGIN_NUM', 'rank_m1_rank_LOGIN_DURATION', 'STUDENTSOURCE_10', 'm2_LOG_DAY', 'PROVINCE_NAME_宁夏回族自治区', 'PROVINCE_NAME_陕西省', 'rank_m1_rank_LM_CLICK_NUM', 'rank_m1_rank_KJ_CLICK_NUM', 'rank_m2_rank_LOGIN_DURATION', 'rank_m5_rank_LOG_DAY', 'MARRIAGE_其他', 'STUDYMODE_2', 'LCENTERTYPENAME_VIP', 'PROVINCE_NAME_天津市', 'PROVINCE_NAME_甘肃省', 'rank_m3_rank_KJ_CLICK_NUM', 'LCENTERTYPERANK_0.0', 'SEX_2', 'rank_m0_rank_LM_CLICK_NUM', 'rank_m2_rank_KJ_CLICK_NUM', 'rank_m3_rank_LM_CLICK_NUM', 'rank_m5_rank_KJ_CLICK_NUM', 'm0_FIN_JOB_NUM', 'PROVINCE_NAME_重庆市', 'PROVINCE_NAME_青海省', 'rank_m2_rank_LM_CLICK_NUM', 'rank_m4_rank_KJ_CLICK_NUM', 'rank_m5_rank_LM_CLICK_NUM', 'ENTRANCETYPE_2', 'm1_FIN_JOB_NUM', 'rank_m4_rank_LM_CLICK_NUM']

# 0.867493140627['age', 'DAY_MEAN_LOGIN_DURATION', 'DAY_MEAN_LOGIN_NUM', 'mean_LOGIN_DURATION', 'm0_LOGIN_DURATION', 'STD_LOGIN_DURATION', 'w_stdLOGIN_DURATION', 'FACTTUITION', 'w_mLOGIN_DURATION', 'DAY_MEAN_KJ_CLICK_NUM', 'STD_LOGIN_NUM', 'w_stdLOGIN_NUM', 'w_stdLOG_DAY', 'STD_LOG_DAY', 'm3_LOGIN_DURATION', 'm4_LOGIN_DURATION', 'm1_LOGIN_DURATION', 'earliestchoosefrom2', 'mean_KJ_CLICK_NUM', 'm0_maxmin_LOGIN_DURATION', 'mean_LOGIN_NUM', 'STD_KJ_CLICK_NUM', 'SEX_1', 'w_mLOGIN_NUM', 'm4_maxmin_LOGIN_DURATION', 'w_stdKJ_CLICK_NUM', 'm0_maxmin_LOGIN_NUM', 'w_mLOG_DAY', 'm2_LOGIN_DURATION', 'm3_maxmin_LOGIN_DURATION', 'm4_maxmin_LOGIN_NUM', 'm3_maxmin_LOGIN_NUM', 'w_mKJ_CLICK_NUM', 'm1_maxmin_LOGIN_NUM', 'mean_LOG_DAY', 'm0_maxmin_LOG_DAY', 'm5_maxmin_LOGIN_NUM', 'LEVELCODE_1', 'm5_LOGIN_DURATION', 'LCENTERTYPERANK_1.0', 'm2_maxmin_LOGIN_NUM', 'm4_maxmin_LOG_DAY', 'm1_maxmin_LOGIN_DURATION', 'm0_LOGIN_NUM', 'm3_maxmin_LOG_DAY', 'm1_LOGIN_NUM', 'm2_maxmin_LOGIN_DURATION', 'm1_KJ_CLICK_NUM', 'm3_LOGIN_NUM', 'm0_KJ_CLICK_NUM', 'DAY_MEAN_LM_CLICK_NUM', 'm5_maxmin_LOGIN_DURATION', 'LCENTERTYPERANK_2.0', 'MARRIAGE_其它', 'm4_LOGIN_NUM', 'LCENTERTYPERANK_3.0', 'm5_maxmin_LOG_DAY', 'm3_KJ_CLICK_NUM', 'm2_maxmin_LOG_DAY', 'm1_maxmin_LOG_DAY', 'PROVINCE_NAME_内蒙古自治区', 'LCENTERTYPERANK_4.0', 'm4_KJ_CLICK_NUM', 'rank_m0_rank_LOG_DAY', 'LCENTERTYPENAME_直属', 'm2_LOGIN_NUM', 'PROVINCE_NAME_江西省', 'LCENTERTYPENAME_普通', 'm0_maxmin_KJ_CLICK_NUM', 'STUDENTSOURCE_5', 'm0_LOG_DAY', 'rank_m4_rank_LOG_DAY', 'STUDENTSOURCE_6', 'm3_maxmin_LM_CLICK_NUM', 'rank_m0_rank_LOGIN_NUM', 'rank_m3_rank_LOGIN_NUM', 'rank_m3_rank_LOG_DAY', 'PROVINCE_NAME_福建省', 'm1_LOG_DAY', 'PROVINCE_NAME_广西壮族自治区', 'm4_maxmin_LM_CLICK_NUM', 'rank_m1_rank_LOG_DAY', 'm5_LOGIN_NUM', 'PROVINCE_NAME_湖南省', 'STD_LM_CLICK_NUM', 'm4_LOG_DAY', 'm3_LOG_DAY', 'm4_LM_CLICK_NUM', 'w_stdLM_CLICK_NUM', 'm5_maxmin_LM_CLICK_NUM', 'rank_m4_rank_LOGIN_NUM', 'PROVINCE_NAME_山东省', 'PROVINCE_NAME_山西省', 'MARRIAGE_已婚', 'mean_LM_CLICK_NUM', 'rank_m1_rank_LOGIN_NUM', 'rank_m2_rank_LOG_DAY', 'm1_maxmin_KJ_CLICK_NUM', 'm4_maxmin_KJ_CLICK_NUM', 'm2_KJ_CLICK_NUM', 'w_mLM_CLICK_NUM', 'MARRIAGE_未婚', 'rank_m0_rank_LOGIN_DURATION', 'rank_m4_rank_LOGIN_DURATION', 'PROVINCE_NAME_河北省', 'PROVINCE_NAME_河南省', 'rank_m3_rank_LOGIN_DURATION', 'rank_m5_rank_LOGIN_NUM', 'rank_m5_rank_LOG_DAY', 'm3_maxmin_KJ_CLICK_NUM', 'm1_LM_CLICK_NUM', 'SEX_2', 'PROVINCE_NAME_吉林省', 'm1_maxmin_LM_CLICK_NUM', 'rank_m2_rank_LOGIN_NUM', 'PROVINCE_NAME_湖北省', 'm3_LM_CLICK_NUM', 'LEVELCODE_3', 'PROVINCE_NAME_辽宁省', 'm0_LM_CLICK_NUM', 'rank_m1_rank_LOGIN_DURATION', 'rank_m5_rank_LOGIN_DURATION', 'PROVINCE_NAME_江苏省', 'm2_maxmin_LM_CLICK_NUM', 'rank_m2_rank_LOGIN_DURATION', 'LCENTERTYPERANK_0.0', 'm2_LOG_DAY', 'm5_maxmin_KJ_CLICK_NUM', 'PROVINCE_NAME_云南省', 'm5_LM_CLICK_NUM', 'rank_m0_rank_KJ_CLICK_NUM', 'm0_maxmin_LM_CLICK_NUM', 'm2_maxmin_KJ_CLICK_NUM', 'm5_LOG_DAY', 'm5_KJ_CLICK_NUM', 'PROVINCE_NAME_广东省', 'rank_m4_rank_LM_CLICK_NUM', 'rank_m3_rank_LM_CLICK_NUM', 'rank_m5_rank_LM_CLICK_NUM', 'MARRIAGE_其他', 'rank_m1_rank_KJ_CLICK_NUM', 'm2_LM_CLICK_NUM', 'PROVINCE_NAME_贵州省', 'rank_m4_rank_KJ_CLICK_NUM', 'LCENTERTYPENAME_VIP', 'rank_m1_rank_LM_CLICK_NUM', 'rank_m3_rank_KJ_CLICK_NUM', 'PROVINCE_NAME_安徽省', 'ENTRANCETYPE_1', 'PROVINCE_NAME_新疆维吾尔自治区', 'PROVINCE_NAME_黑龙江省', 'rank_m0_rank_LM_CLICK_NUM', 'STUDYMODE_1', 'PROVINCE_NAME_浙江省', 'rank_m2_rank_KJ_CLICK_NUM', 'PROVINCE_NAME_陕西省', 'rank_m5_rank_KJ_CLICK_NUM', 'rank_m2_rank_LM_CLICK_NUM', 'PROVINCE_NAME_北京市', 'STUDENTSOURCE_7', 'PROVINCE_NAME_甘肃省', 'ENTRANCETYPE_2', 'STUDYMODE_2', 'm1_FIN_JOB_NUM', 'PROVINCE_NAME_重庆市', 'STUDENTSOURCE_10', 'PROVINCE_NAME_四川省', 'PROVINCE_NAME_天津市', 'm2_FIN_JOB_NUM', 'm0_FIN_JOB_NUM', 'm4_FIN_JOB_NUM', 'PROVINCE_NAME_上海市', 'PROVINCE_NAME_青海省']
