from random import random

import pandas as pd
import torch

import matplotlib.pyplot as plt
# from pandas import pd
from torch.optim import optimizer
from torch.utils.data import SubsetRandomSampler,SequentialSampler
import numpy as np
from dataloader_w import wDataset, device
from TransformerModel import TransformerModel
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True



dataset = wDataset()
dataset_size=dataset.len
setup_seed(20)

bs = 64
nvars = 10
seq_len = 7
c_out = 1
# xb = torch.rand(bs, nvars, seq_len)

net = TransformerModel(nvars, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation='gelu', n_layers=3)

shuffle_dataset=False
validation_split = .8
random_seed = 42
batch_size=16
split = int(np.floor(validation_split * dataset_size))
indices = list(range(dataset.len))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]

train_sampler = SequentialSampler(train_indices)
valid_sampler = SequentialSampler(val_indices)
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)


loss_func = torch.nn.L1Loss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.0001)


def train(epoch=None, ):
    net.train()
    y1 = []
    y2 = []
    net_epoch1="0"
    min_testloss = 100000
    for epoch in range(epoch):
        train_loss = []
        test_loss = []
        for data in train_loader:
            onebatch,onebatchlables=data
            onebatch=onebatch.type(torch.FloatTensor).to("cpu")
            onebatchlables =onebatchlables.type(torch.FloatTensor).to("cpu")
            output=net(onebatch)
            loss = loss_func(output, onebatchlables)  # cross entropy loss
            train_loss.append(loss.sum())
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()

        net.eval()
        with torch.no_grad():
            for validata in test_loader:
                onetestbatch,onetestbatchlables=validata
                onetestbatch=onetestbatch.type(torch.FloatTensor).to("cpu")
                onetestbatchlables=onetestbatchlables.type(torch.FloatTensor).to("cpu")
                test_output=net(onetestbatch)

                testloss =loss_func(test_output,onetestbatchlables)
                test_loss.append(testloss.sum())
            # print(sum(test_loss))
        y1.append((sum(train_loss)/len(train_loss)).detach().numpy())
        y2.append((sum(test_loss)/len(test_loss)).detach().numpy())


        if y2[-1].item()<min_testloss:
            min_testloss=y2[-1].item()
            torch.save(net, "goodnet.pt")

            net_epoch1=str(epoch)
            print("epoch:{},train loss:{},test loss:{}".format(net_epoch1,y1[-1],y2[-1]))

    epochs = range(1, len(y1) + 1)

    loss_data = {'loss':y1,'val_loss':y2}
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv('loss.csv',index=False)

    fig1 = plt.figure(figsize=(8, 5), dpi=300)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    fontdict = {'weight': 'bold', 'size': 13, "family": 'Times New Roman'}
    plt.plot(epochs, y1, label='Training_set_loss',color='blue')
    plt.plot(epochs, y2, label='Validation_set_loss',color='red')
    plt.legend()
    plt.xlabel('Epochs',fontdict=fontdict)
    plt.ylabel('Loss',fontdict=fontdict)
    # plt.title('Traning Loss')
    plt.savefig('Traning Loss.png')
    plt.close()
    return y1,y2,net_epoch1


val_pre_list = []
train_pre_list = []
val_real_list = []
train_real_list = []

if __name__ == '__main__':

    train(300)
    # net = torch.load("goodnet.pt")
    # scaler = dataset.scaler
    # real_y = dataset.Y
    # out = net(dataset.X.type(torch.FloatTensor)).detach().numpy()
    # data_length = out.shape[0]
    # split = 0.8
    # point = int(split*data_length)
    # train_y = out[:point]
    # val_y = out[point:]
    # train_y_inv = scaler.inverse_transform(train_y)
    # val_y_inv = scaler.inverse_transform(val_y)
    # real_y_inv = scaler.inverse_transform(real_y)
    # x = np.arange(1,data_length+1,1)
    # x1 = np.arange(1,point+1,1)
    # x2 = np.arange(point+1,data_length+1,1)
    #
    # fig1 = plt.figure(figsize=(8,5),dpi=300)
    # fontdict = {'weight': 'bold', 'size': 13,"family": 'Times New Roman'}
    #
    # fig2 = plt.figure(figsize=(8,5),dpi=200)
    # plt.plot(x,real_y, label='Original_data', color='blue')
    # plt.plot(x1,train_y, label='Training_set_prediction', color='red')
    # plt.plot(x2, val_y, label='Validation_set_prediction',color='green')
    # plt.xlabel('Time(day)',fontdict=fontdict)
    # plt.ylabel('Liquid(Normalized)',fontdict=fontdict)
    # plt.legend()
    # plt.savefig('predict.png')
    # plt.close()
    #
    # plt.plot(x,real_y_inv, label='Original_data', color='blue')
    # plt.plot(x1,train_y_inv, label='Training_set_prediction', color='red')
    # plt.plot(x2, val_y_inv, label='Validation_set_prediction',color='green')
    # plt.xlabel('Time(day)',fontdict=fontdict)
    # plt.ylabel('Liquid(t/d)',fontdict=fontdict)
    # plt.legend()
    # plt.savefig('predict_inverse.png')
    # plt.close()


    # x = np.arange(0,len(out),1)
    # plt.plot(x,out.detach().numpy())
    # plt.show()
    # for train in train_loader:
    #     onetrainbatch,onetrainbatchlables = train
    #     onetrainbatch = onetrainbatch.type(torch.FloatTensor)
    #     onetrainbatchlables = onetrainbatchlables.type(torch.FloatTensor)
    #     train_output = net(onetrainbatch).detach().numpy()
    #     train_pre_list.append(train_output)
    #     train_real_list.append(onetrainbatchlables.detach().numpy())
    # for validata in test_loader:
    #
    #     onetestbatch, onetestbatchlables = validata
    #     onetestbatch = onetestbatch.type(torch.FloatTensor)
    #     onetestbatchlables = onetestbatchlables.type(torch.FloatTensor)
    #     test_output = net(onetestbatch).detach().numpy()
    #     val_pre_list.append(test_output)
    #     val_real_list.append(onetestbatchlables.detach().numpy())



# val_arr = np.vstack(val_pre_list)
# train_arr = np.vstack(train_pre_list)

# val_real_arr = np.vstack(val_real_list)
# train_real_arr = np.vstack(train_real_list)
# real = np.vstack([train_real_arr,val_real_arr])
# total_length = len(val_real_arr)+len(train_real_arr)
# x,y=dataset.data_process()
# x_ = np.arange(1,len(y)+1,1)
# x1 = np.arange(1,len(train_arr)+1,1)
# x2 = np.arange(len(train_arr)+1,total_length+1,1)
# plt.plot(x_,y,label='real')
# plt.plot(x1,train_arr,label='train')
# plt.plot(x2,val_arr,label='val')
# plt.legend()
# plt.show()
