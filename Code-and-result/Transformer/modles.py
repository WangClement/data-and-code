#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1.20
@File ：MAE.py
@Author ：ts
@Date ：2022/2/11 21:17
'''
import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,input_dim,num_hiddens,num_context):
        super(Encoder, self).__init__()
        # self.embedding=nn.Linear(input_dim,embed_dim)
        self.embedding=nn.Linear(input_dim,num_hiddens)
        self.fc=nn.Linear(num_hiddens,num_context)
        self.activation=nn.ReLU()

    def forward(self,x):
        x=self.embedding(x)
        x=self.activation(x)
        output=self.fc(x)
        return output

class Decoder(nn.Module):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, input_dim, num_hiddens,num_context,output_dim
                 ):
        super(Decoder, self).__init__()
        # self.embedding = nn.Linear(input_dim, embed_dim)
        self.embedding=nn.Linear(input_dim,num_hiddens)
        self.dense = nn.Linear(num_hiddens+num_context, num_context)
        self.dense2 = nn.Linear(num_context, output_dim)
        self.activation=nn.ReLU()


    def forward(self, X, context):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        X=torch.cat([X,context],dim=1)
        out=self.activation(self.dense(X))
        out = self.activation(self.dense2(out))
        return out



class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.
    Defined in :numref:`sec_encoder-decoder`"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)

        return self.decoder(dec_X, enc_outputs)
if __name__ == '__main__':
    encoder=Encoder(400,512,256)
    tensor=torch.ones((187,400))

    decoder=Decoder(400,512,256,30)
    net=EncoderDecoder(encoder,decoder)
    out=net(tensor,tensor)
    print(out.shape)

