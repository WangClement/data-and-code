# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/108_models.TransformerModel.ipynb (unless otherwise specified).


# from layers import *
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

# Cell
class TransformerModel(nn.Module):
    def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation="relu", n_layers=1):
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
            d_model: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.

        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
            """
        # self.permute = Permute(1, 0, 2)
        super(TransformerModel, self).__init__()
        self.inlinear = nn.Linear(c_in, d_model)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout, activation=activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, norm=encoder_norm)
        # self.transpose = Transpose(1, 0)
        # self.max = Max(1)
        self.outlinear = nn.Linear(d_model, c_out)

    def forward(self,x):
        x=x.permute(1,0,2)# bs x seq_len x nvars -> seq_len x bs x nvars
        x = self.inlinear(x) # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1,0) # seq_len x bs x d_model -> bs x seq_len x d_model
        x =x.max(1)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        return x
if __name__ == '__main__':
    tensor=torch.ones(64,200,15)
    bs = 64
    nvars = 15
    seq_len = 200
    c_out = 1
    # xb = torch.rand(bs, nvars, seq_len)

    net = TransformerModel(nvars, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation='gelu', n_layers=3)
    out=net(tensor)
    print(out.shape)