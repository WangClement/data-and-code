import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import SimpleRNN,Dropout,BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from icecream.icecream import ic
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.vis_utils import plot_model
from matplotlib.ticker import FuncFormatter

# 设置西文字体为新罗马字体
from matplotlib import rcParams
def to_percent(temp, position):
    return '%.1f'%(100*temp) + '%'

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "font.size": 13,
    "font.weight":'bold'
    #     "mathtext.fontset":'stix',
}
rcParams.update(config)
ic.enable()
#设置数据的时间窗口

#设置gpu训练
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpu0], "GPU")


def model_training(data_dic,units1=128,units2=64,epochs=300,batch_size=100):
    x_train, y_train, x_val, y_val, y_real = data_dic['x_train'], data_dic['y_train'], data_dic['x_val'], \
                                             data_dic['y_val'], data_dic['y_real']
    model = models.Sequential()
    model.add(SimpleRNN(units=units1, input_shape=((x_train.shape[1],
                                                x_train.shape[2])),
                       return_sequences=True,
                       kernel_initializer='random_uniform')
                  )

    model.add(SimpleRNN(units=units2))
    model.add(layers.Dense(1))

    model.compile(loss='mae',
                  optimizer='adam'
                  )
    model.summary()

    history = model.fit(x_train, y_train,
                            validation_data=(x_val, y_val),
                            epochs=epochs, batch_size=batch_size,
                            )
    history_dic = history.history
    loss = history_dic['loss']
    val_loss = history_dic['val_loss']
    epochs = range(1, len(loss) + 1)

    fig1 = plt.figure(figsize=(8, 5), dpi=300)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    fontdict = {'weight': 'bold', 'size': 13, "family": 'Times New Roman'}
    plt.plot(epochs, loss, label='Training_set_loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation_set_loss', color='red')
    plt.legend()
    plt.xlabel('Epochs', fontdict=fontdict)
    plt.ylabel('Loss', fontdict=fontdict)
    # plt.title('Traning Loss',fontdict=fontdict)
    plt.savefig('Traning Loss.png')
    plt.close()
    model.save('my_model.h5')
    history_df = pd.DataFrame(history_dic)
    history_df.to_csv('loss.csv', index=False)

"""
func:数据预处理(空值填充、归一化、滑窗、验证集分割)
para:window 设置滑窗长度 start,end设置数据集起始,split_point设置验证集分割点
return:data_dic scaler_dic
"""
def data_process(window = 7,start=100,end=1200,split_point=0.8):
    data = pd.read_excel('C:\pythonproject\yongjin_training\永进油田开发日数据.xls',
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
    x_data, y_data = np.zeros((data_length, window, dem_x)), np.zeros((data_length, 1, dem_y))
    for i in range(data_x_scaled.shape[0] - window):
        x = data_x_scaled[i:i + window]
        x_data[i] = x
        y = data_y_scaled[i + window]
        y_data[i] = y
    ic(x_data.shape, y_data.shape)

    #训练集验证集分割
    point = int(data_length*split_point)
    x_train = x_data[:point]
    y_train = y_data[:point]
    x_val = x_data[point:]
    y_val = y_data[point:]
    ic(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    y_real = np.vstack([y_train, y_val])

    data_dic = {'x_train':x_train,'y_train':y_train,
                'x_val':x_val,'y_val':y_val,'y_real':y_real}
    scaler_dic = {'scaler_x':scaler_x,'scaler_y':scaler_y}

    return data_dic,scaler_dic


def result_view(data_dic,scaler_dic):
    x_train, y_train, x_val, y_val, y_real = data_dic['x_train'], data_dic['y_train'], data_dic['x_val'], \
                                             data_dic['y_val'], data_dic['y_real']
    scaler_y = scaler_dic['scaler_y']

    # 加载模型，并预测
    model = models.load_model('my_model.h5')
    y_train_predict = model.predict(x_train).reshape((len(x_train), 1))
    y_val_predict = model.predict(x_val).reshape((len(x_val), 1))

    # 数据还原为归一化前的尺度上
    y_train_predict_inverse = scaler_y.inverse_transform(y_train_predict)
    y_val_predict_inverse = scaler_y.inverse_transform(y_val_predict)
    y_real_invers = scaler_y.inverse_transform(y_real.reshape((len(y_real), 1)))

    # 设置x轴
    step_train = np.arange(1, len(y_train) + 1, 1)
    step_val = np.arange(len(y_train) + 1, len(y_real) + 1, 1)
    step_orignal = np.arange(1, len(y_real) + 1, 1)

    fig1 = plt.figure(figsize=(8, 5), dpi=300)
    fontdict = {'weight': 'bold', 'size': 13, "family": 'Times New Roman'}
    bwith = 1  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    plt.plot(step_orignal, y_real.reshape((len(y_real), 1)), label='Original_data', color='blue')
    plt.plot(step_train, y_train_predict, label='Training_set_prediction', color='red')
    plt.plot(step_val, y_val_predict, label='Validation_set_prediction', color='green')
    plt.xlabel('Time(day)', fontdict=fontdict)
    plt.ylabel('Liquid(Normalized)', fontdict=fontdict)
    plt.legend()
    fig1.savefig('predict.png')
    plt.close()

    # 作图
    fig2 = plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(step_orignal, y_real_invers, label='Original_data', color='blue')
    plt.plot(step_train, y_train_predict_inverse, label='Training_set_prediction', color='red')
    plt.plot(step_val, y_val_predict_inverse, label='Validation_set_prediction', color='green')
    plt.xlabel('Time(day)', fontdict=fontdict)
    plt.ylabel('Liquid(t/d)', fontdict=fontdict)
    plt.legend()
    fig2.savefig('predict_inverse.png')
    plt.close()
    plot_model(model, to_file='model.png')


data_dic,scaler_dic = data_process()
model_training(data_dic,epochs=300)
result_view(data_dic,scaler_dic)