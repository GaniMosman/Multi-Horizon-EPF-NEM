import math
import torch
import torch.nn as nn


# -------------------------
# LSTM and BiLSTM model
# -------------------------
class LSTM(nn.Module):
    def __init__(self, num_features, output_seq_len, output_features = 1, hidden_units = 128,
                    num_layers = 1, dropout_rate = 0, bidirectional = False):

        super().__init__()
        self.num_features = num_features
        self.hidden_units =  int(hidden_units)    
        self.num_layers = int(num_layers)      
        self.output_seq_len = output_seq_len
        self.output_features = output_features
        self.dropout = dropout_rate if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional = self.bidirectional
        )
       
        self.linear = nn.Linear(in_features=self.hidden_units*self.num_directions, out_features=self.output_features)

    def forward(self, x):
        batch_size = x.shape[0]

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_units, device=x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_units, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -self.output_seq_len:, :]

        output = self.linear(out)

        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_seq_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.output_seq_len, 1))
       
        return output.squeeze(-1)

# -------------------------
# CNN-LSTM model
# -------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, num_features, output_seq_len, output_features = 1, hidden_units = 128, num_layers = 1,
                 dropout_rate = 0, cnn_channels = 128, kernel_size = 5, bidirectional = False):
        
        super().__init__()
        self.num_features = num_features
        self.cnn_channels = int(cnn_channels)
        self.kernel_size = int(kernel_size)
        self.hidden_units = int(hidden_units)     
        self.num_layers = int(num_layers)  
        self.output_seq_len = output_seq_len
        self.output_features = output_features
        self.dropout = dropout_rate if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.cnn = nn.Conv1d(
            in_channels=self.num_features, 
            out_channels=self.cnn_channels, 
            kernel_size=self.kernel_size, 
            padding=self.kernel_size//2 
        )
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_channels,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        
        self.linear = nn.Linear(
            in_features=self.hidden_units * self.num_directions, 
            out_features=self.output_features
        )

    def forward(self, x):
        batch_size = x.shape[0]

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        
        x_cnn = x.permute(0, 2, 1) 
        x_cnn = self.relu(self.cnn(x_cnn))
        
        x_lstm = x_cnn.permute(0, 2, 1)

        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_units,
            device=x.device
            )
        c0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_units,
            device=x.device
            )

        out, _ = self.lstm(x_lstm, (h0, c0))
        out = out[:, -self.output_seq_len:, :] 

        output = self.linear(out)

        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_seq_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.output_seq_len, 1))

        return output.squeeze(-1)


# -------------------------
# Transformer model
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class Transformer(nn.Module):
    def __init__(self,
        num_features,
        output_seq_len,
        d_model = 128,
        num_heads = 8,
        num_layers = 1,
        output_features = 1,
        dropout_rate = 0
        ):
        super().__init__()

        self.num_features = num_features
        self.output_seq_len = output_seq_len
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.output_features = output_features
        self.dropout = dropout_rate

        self.input_embedding = nn.Linear(self.num_features, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward = 4 * self.d_model,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.output_projection = nn.Linear(self.d_model, self.output_features)

    def forward(self, x):

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)  
        encoder_output = self.transformer_encoder(x)  
        selected_output = encoder_output[:, -self.output_seq_len:, :] 
        output = self.output_projection(selected_output)

        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_seq_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.output_seq_len, 1))

        return output.squeeze(-1)  