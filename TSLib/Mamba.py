import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba as mamba
from Modules.TSLib.layers.Embed import DataEmbedding


### DataEmbedding and Normalisation is not part of the model design
## Need to tune d_model, d_state(max 16)

class Mamba(nn.Module):
    
    def __init__(self, num_features, pred_len, d_model=128, d_state=16, d_conv=4,
                 expand=2, embed = "timeF",  freq ='h', dropout=0, output_features=1):
        super().__init__()
        self.enc_in = int(num_features)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = d_conv
        self.expand = expand
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.c_out = output_features
        
        self.embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)

        self.mamba = mamba(
            d_model = self.d_model,
            d_state = self.d_state,
            d_conv = self.d_conv,
            expand = self.expand,
        )

        self.out_layer = nn.Linear(self.d_model, self.c_out, bias=False)

    def forward(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)

        x = x[:, -self.pred_len:, :]

        x_out = self.out_layer(x)

        target_std = std_enc[:, :, [0]]  
        target_mean = mean_enc[:, :, [0]]
        x_out = x_out * target_std + target_mean
        
        x_out = x_out.squeeze(-1)

        return x_out