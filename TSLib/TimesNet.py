import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from Modules.TSLib.layers.Embed import DataEmbedding
from Modules.TSLib.layers.Conv_Blocks import Inception_Block_V1



def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len,  top_k, d_model, d_ff, num_kernels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)  # 2D conv
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


# Initial LR 10−4 used in original research article
# k/top_k is set to 5 in riginal research article
# d_model search between [32 512]
# loss MSE (L2)
# Need to tune d_model, d_ff

class TimesNet(nn.Module):

    def __init__(self, num_features, seq_len, pred_len, d_model= 128, d_ff = 128, e_layers=2,
                  num_kernels=3, top_k = 5, embed = "timeF", freq = 'h', 
                 dropout = 0, output_features = 1):
        super().__init__()
        self.enc_in = int(num_features)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.e_layers = int(e_layers)
        self.d_model = int(d_model)
        self.d_ff = int(d_model * 4)
        self.c_out = output_features
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.top_k = top_k
        self.num_kernels = num_kernels

        self.enc_embedding = DataEmbedding(
            self.enc_in, self.d_model, self.embed, self.freq, self.dropout
        )
        configs = [self.seq_len, self.pred_len, self.top_k, self.d_model, self.d_ff, self.num_kernels]
        self.model = nn.ModuleList([TimesBlock(*configs) for _ in range(self.e_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesNet layers
        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Projection
        dec_out = self.projection(enc_out)

        #De-Normalization
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))

        dec_out = dec_out[:, -self.pred_len:, 0:1].squeeze(-1)

        return dec_out