import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.TSLib.layers.Transformer_EncDec import Encoder, EncoderLayer
from Modules.TSLib.layers.SelfAttention_Family import FullAttention, AttentionLayer
from Modules.TSLib.layers.Embed import DataEmbedding_inverted


"""
Original work
Initial learning rate in
{ 10− 3
, 5× 10− 4
, 10− 4
} and L2 loss for the model optimization.
Dimension of series representations (d_model) is set from { 256, 512}
"""

# Need tune d_model, d_ff
class iTransformer(nn.Module):
    
    def __init__(self, seq_len, pred_len, d_model = 128, d_ff = 128, e_layers=2, 
                 embed ="timeF", freq= 'h', dropout=0, factor=1, n_heads=8, 
                 activation= "gelu"):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.d_ff = int(d_model * 4) 
        self.e_layers = int(e_layers)
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.factor = factor
        self.n_heads = n_heads
        self.activation = activation
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=False), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)
        
    def forward(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out[:, -self.pred_len:, 0:1].squeeze(-1)
