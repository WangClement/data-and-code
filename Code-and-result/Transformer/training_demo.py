import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


"""
func:数据预处理(空值填充、归一化、滑窗、验证集分割)
para:window 设置滑窗长度 start,end设置数据集起始,split_point设置验证集分割点
"""
def data_process(window = 7,start=100,end=1200,split_point=0.8):
    data = pd.read_excel('永进油田开发日数据.xls',
                         sheet_name='永1-平1')

    well_open = []
    for i in range(data.shape[0]):
        hours = data.iloc[i]['生产时间']
        if hours == 0:
            well_open.append(0)
        else:
            well_open.append(1)

    data['开关井'] = well_open
    # 选取训练所需字段并截取前1000行数据
    data_training = data[['生产时间', '油压', '套压', '回压',
                          '冲程', '冲次', '日产油',
                          '含水', '井口温度', '开关井', '日产液']].iloc[start:end + window]

    # 空值填充
    data_training.fillna(method='backfill', inplace=True)

    # 将数据归一化
    # [:-1]列为训练输入 [-1]列为输出
    x_columns = ['生产时间', '油压', '套压', '回压', '冲程',
                 '冲次', '日产油', '含水', '井口温度', '开关井']
    y_columns = ['日产液']
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    data_x_scaled = scaler_x.fit_transform(data_training[x_columns])
    data_y_scaled = scaler_y.fit_transform(data_training[y_columns])

    # 设置输入数据的特征维度和输出数据的维度以及数据集长度
    dem_x = data_x_scaled.shape[1]
    dem_y = data_y_scaled.shape[1]
    data_length = data_x_scaled.shape[0] - window

    # 对数据进行滑窗
    x_data, y_data = np.zeros((data_length, window, dem_x)), np.zeros((data_length, dem_y))
    for i in range(data_x_scaled.shape[0] - window):
        x = data_x_scaled[i:i + window]
        x_data[i] = x
        y = data_y_scaled[i + window]
        y_data[i] = y


    #训练集验证集分割
    point = int(data_length*split_point)
    x_train = x_data[:point]
    y_train = y_data[:point]
    x_val = x_data[point:]
    y_val = y_data[point:]

    return x_train,y_train,x_val,y_val

x_train,y_train,x_val,y_val = data_process()
print(x_train,y_train,x_val) #(1100,7,10)