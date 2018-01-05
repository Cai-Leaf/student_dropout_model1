import pandas as pd

data = pd.read_csv('../data_feature/feature4.csv')
pos_label = data[data.tz_students == 1]
print(len(data), len(pos_label), len(pos_label)/len(data))

