import pandas as pd

data = pd.read_csv('../data_feature/feature4.csv')
data = data[['']]

print(len(data[(data.avgbxgetfinalgrade_corr < 40)]))
