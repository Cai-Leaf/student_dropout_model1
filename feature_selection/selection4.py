import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt
import time
from sklearn import model_selection


data = pd.read_csv('feature.csv')
data_y = data['tz_students'].values
data_y = 1-data_y
data = data.drop(['STUDENTCODE', 'tz_students'], axis=1)

# lightgbm
clf = LGBMClassifier(num_leaves=40, learning_rate=0.05, max_depth=20, n_estimators=300, subsample=0.8,
                    colsample_bytree=1, min_child_weight=1)

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

#  做出基本分
tmp_data = data[name_list]
start = time.time()
# 交叉验证
score_name = 'roc_auc'
tmp_clf = LGBMClassifier(num_leaves=40, learning_rate=0.05, max_depth=20, n_estimators=300, subsample=0.8,
                         colsample_bytree=1, min_child_weight=1)
score = model_selection.cross_val_score(estimator=tmp_clf, X=tmp_data, y=data_y,
                                        cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
                                        scoring=score_name, groups=data_y)
bais_s = sum(score)/len(score)
print(bais_s)

# 特征选择
i = len(name_list)-1
while i > 9:
    save_feature = name_list[i]
    del name_list[i]
    tmp_data = data[name_list]
    # 交叉验证
    score_name = 'roc_auc'
    tmp_clf = LGBMClassifier(num_leaves=40, learning_rate=0.05, max_depth=20, n_estimators=300, subsample=0.8,
                             colsample_bytree=1, min_child_weight=1)
    score = model_selection.cross_val_score(estimator=tmp_clf, X=tmp_data, y=data_y,
                                            cv=model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
                                            scoring=score_name, groups=data_y)
    mean_score = sum(score)/len(score)
    print(i, score)
    print(i, score_name + ':', mean_score)
    if mean_score < bais_s:
        name_list.append(save_feature)
        print('save', save_feature)
    else:
        bais_s = mean_score
        print('del', save_feature)
    i -= 1
    print('feature_num:', len(name_list), 'base score', bais_s)
    print('_________________________________________________________________________________________________')

print(name_list)
# 选好的特征
# ['DAY_MEAN_KJ_CLICK_NUM', 'DAY_MEAN_LM_CLICK_NUM', 'DAY_MEAN_LOGIN_DURATION', 'DAY_MEAN_LOGIN_NUM', 'FACTTUITION',
#  'LCENTERTYPENAME_直属', 'LCENTERTYPERANK_1.0', 'LCENTERTYPERANK_2.0', 'LCENTERTYPERANK_3.0', 'LCENTERTYPERANK_4.0',
#  'MARRIAGE_其它', 'MARRIAGE_未婚', 'PROVINCE_NAME_云南省', 'PROVINCE_NAME_内蒙古自治区', 'PROVINCE_NAME_山东省',
#  'PROVINCE_NAME_山西省', 'PROVINCE_NAME_广东省', 'PROVINCE_NAME_广西壮族自治区', 'PROVINCE_NAME_江西省',
#  'PROVINCE_NAME_河南省', 'PROVINCE_NAME_湖北省', 'PROVINCE_NAME_福建省', 'PROVINCE_NAME_辽宁省',
#  'SEX_1', 'STD_LM_CLICK_NUM', 'STD_LOGIN_DURATION', 'STD_LOGIN_NUM', 'STD_LOG_DAY', 'STUDENTSOURCE_5', 'age',
#  'earliestchoosefrom2', 'm0_KJ_CLICK_NUM', 'm0_LOGIN_DURATION', 'm0_LOGIN_NUM', 'm0_maxmin_KJ_CLICK_NUM',
#  'm0_maxmin_LOGIN_DURATION', 'm0_maxmin_LOGIN_NUM', 'm0_maxmin_LOG_DAY', 'm1_KJ_CLICK_NUM', 'm1_LOGIN_DURATION',
#  'm1_LOGIN_NUM', 'm1_maxmin_LOGIN_DURATION', 'm1_maxmin_LOGIN_NUM', 'm1_maxmin_LOG_DAY', 'm2_LOGIN_DURATION',
#  'm2_LOGIN_NUM', 'm2_maxmin_LOGIN_DURATION', 'm2_maxmin_LOGIN_NUM', 'm2_maxmin_LOG_DAY', 'm3_KJ_CLICK_NUM',
#  'm3_LOGIN_DURATION', 'm3_LOGIN_NUM', 'm3_maxmin_LOGIN_DURATION', 'm3_maxmin_LOGIN_NUM', 'm3_maxmin_LOG_DAY',
#  'm4_KJ_CLICK_NUM', 'm4_LM_CLICK_NUM', 'm4_LOGIN_DURATION', 'm4_LOGIN_NUM', 'm4_maxmin_LOGIN_DURATION',
#  'm4_maxmin_LOGIN_NUM', 'm4_maxmin_LOG_DAY', 'm5_LOGIN_DURATION', 'm5_LOGIN_NUM', 'm5_maxmin_LOGIN_DURATION',
#  'm5_maxmin_LOGIN_NUM', 'm5_maxmin_LOG_DAY', 'mean_KJ_CLICK_NUM', 'mean_LOGIN_DURATION', 'mean_LOGIN_NUM',
#  'mean_LOG_DAY', 'rank_m0_rank_LOG_DAY', 'rank_m1_rank_LOGIN_NUM', 'rank_m3_rank_LOGIN_DURATION',
#  'rank_m5_rank_LOGIN_NUM', 'w_mKJ_CLICK_NUM', 'w_mLOGIN_DURATION', 'w_mLOGIN_NUM', 'w_mLOG_DAY', 'w_stdKJ_CLICK_NUM',
#  'w_stdLOGIN_DURATION', 'w_stdLOGIN_NUM', 'w_stdLOG_DAY']