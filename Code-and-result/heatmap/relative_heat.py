import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel('data.xlsx')
print(data.columns)

data.drop(['No.','Well', 'Well Completion'],inplace=True,axis=1)
data_array = data.to_numpy()

scaler = MinMaxScaler()
data_array = scaler.fit_transform(data_array)

def person(arr1,arr2):
    std1 = np.std(arr1)
    std2 = np.std(arr2)

    delta1 = arr1 - np.mean(arr1)
    delta2 = arr2 - np.mean(arr2)

    cov = np.mean(delta1*delta2)
    return cov/(std1*std2)

col = []
for i in range(data_array.shape[1]):
    row = []
    for j in range(data_array.shape[1]):
        vec1 = data_array[:,i]
        vec2 = data_array[:,j]
        p = round(person(vec1,vec2),2)
        row.append(p)
    col.append(row)
mat = np.array(col)
mat = pd.DataFrame(mat,columns=data.columns,index=data.columns)
plt.rcParams['font.sans-serif'] = ['simsun']
plt.figure(figsize=(15, 15))

sns.heatmap(data=mat,square=True,cmap='RdYlBu',
                annot=True,vmin=-1,vmax=1,
                annot_kws={"fontsize":10,'weight':'bold'}
                )

plt.savefig('heatmap.svg',dpi=600,format='svg')
plt.close()