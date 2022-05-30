import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
# 设置西文字体为新罗马字体
from matplotlib import rcParams
def to_percent(temp, position):
    return '%.1f'%(100*temp) + '%'

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "font.size": 15,
    "font.weight":'bold'
    #     "mathtext.fontset":'stix',
}
rcParams.update(config)


data = pd.read_excel('data.xlsx')
print(data.columns)

data.drop(['No.','Well', 'Well Completion'],inplace=True,axis=1)
data_array = data.to_numpy()

scaler = MinMaxScaler()
data_array = scaler.fit_transform(data_array)
print(data_array)

pca=PCA()
pca.fit(data_array)
pca_data=pca.transform(data_array)
per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
fig = plt.figure(figsize=(10,8),dpi=200)
ax = fig.add_subplot(1,1,1)

exp_var_ratio = []
for n in range(1,13):
    pca = PCA(n_components = n)
    pca.fit(pca_data)
    pca.transform(pca_data)
    exp_var_ratio.append(sum(pca.explained_variance_ratio_)*100)

plt.plot(range(1,13), exp_var_ratio, marker = 'o', markerfacecolor = 'red', markersize = 6,color='red')
for i,j in zip(range(1,13),exp_var_ratio):
    value = '%.2f'%(j)
    plt.text(i,j+.5,value, fontsize=15,rotation=-30)

fontdict = {'weight': 'bold', 'size': 15, "family": 'Times New Roman'}
plt.ylabel('Percentage of Explained Variance(%)',fontdict=fontdict)
plt.xlabel('Principal Component',fontdict=fontdict)
plt.xticks(rotation=90) # 旋转90度)

labels = ['PC' + str(x) for x in range(1, 13+1)]
plt.bar(x=range(1,13+1), height=per_var[:13], tick_label=labels,color='blue')
plt.ylim(0,110)
plt.savefig('碎石图.png')