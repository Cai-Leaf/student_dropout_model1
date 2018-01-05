import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv('../data_origin/student_msg.csv', header=0)
data = data[['STUDENTCODE', 'PROVINCE_NAME']]
data = data.set_index('STUDENTCODE')

data = pd.get_dummies(data)

n = 8
pca = PCA(n_components=n)
new_val = pca.fit_transform(data)
new_name = ['PROVINCE_NAME_'+str(i) for i in range(n)]
new_val = pd.DataFrame(new_val, columns=new_name, index=data.index)
new_val['STUDENTCODE'] = new_val.index

new_val['STUDENTCODE'] = new_val.index
data2 = pd.read_csv('../data_feature/feature6.csv', header=0)
data = pd.merge(data2, new_val, on='STUDENTCODE', how='left')
print(data.info())
data.to_csv('../data_feature/feature7_2.csv', index=False, encoding='utf-8')



# data['STUDENTCODE'] = data.index
# data2 = pd.read_csv('../data_feature/feature6.csv', header=0)
# data = pd.merge(data2, data, on='STUDENTCODE', how='left')
# data.to_csv('../data_feature/feature7_1.csv', index=False, encoding='utf-8')



