#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：fault
@File ：fualtDataset.py
@Author ：ts
@Date ：2022/2/18 12:02
'''
import os
import scipy.io as scio
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
###
from torch.utils.data.dataset import T_co
from sklearn.preprocessing import MinMaxScaler

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# scalerdf=pd.read_csv("19-3allwell.csv")
# dfcol=[ 'OILPRESSURE', 'CASINGPRESSURE', 'BACKPRESSURE',
#        'PUMPINLETPRESSURE', 'PUMPOUTPRESSURE', 'PUMPINLETTEMPERTURE',
#        'MOTORTEMPERTURE', 'CURRENTS', 'VOLTAGE', 'FREQUENCY_POWER', 'CREEPAGE',
#        'ChokeDiameter', 'VIB', 'MOTORPOWER', 'RUNTIME']
robust=preprocessing.RobustScaler()


class wDataset(Dataset):

    def __init__(self) -> None:
        self.data=self.data_process()
        self.features = self.data[0]
        self.lables = self.data[1]
        self.scaler = self.data[2]
        self.X,self.Y=self.loaddata()
        self.len = len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index],self.Y[index]

    def loaddata(self):
        X=torch.from_numpy(self.features)
        Y=torch.from_numpy(self.lables)


        return X,Y

    def data_process(self,window=7, start=100, end=1200, split_point=0.8):
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
                              '含水', '井口温度', '开关井', '日产液']].iloc[start:(end + window)]

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
        return x_data,y_data,scaler_y




#
if __name__ == '__main__':
    dataset=wDataset()
    print(dataset.len)

