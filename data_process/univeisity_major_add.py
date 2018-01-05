import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv('../data_origin/student_msg.csv', header=0)


university = data[['STUDENTCODE', 'UNIVERSITYCODE']]
university['UNIVERSITYCODE'] = university['UNIVERSITYCODE'].astype(str)
university = university.set_index('STUDENTCODE')
university = pd.get_dummies(university)

major = data[['STUDENTCODE', '小类名称']]
major = major.set_index('STUDENTCODE')
major = pd.get_dummies(major)

data = data[['STUDENTCODE']]

for u_code in university.columns:
    for m_name in major.columns:
        data[u_code+'_'+m_name] = university[u_code].values * major[m_name].values

data = data.set_index('STUDENTCODE')
n = 30
pca = PCA(n_components=n)
new_val = pca.fit_transform(data)
new_name = ['U_M_'+str(i) for i in range(n)]
data = pd.DataFrame(new_val, columns=new_name, index=data.index)
data['STUDENTCODE'] = data.index

data2 = pd.read_csv('../data_feature/feature7_1.csv', header=0)
data = pd.merge(data2, data, on='STUDENTCODE', how='left')
print(data.info())
data.to_csv('../data_feature/feature8.csv', index=False, encoding='utf-8')
