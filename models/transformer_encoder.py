import torch
import torch.nn as nn
import math
from config import TRANSFORMER_EMBED_DIM, TRANSFORMER_NUM_HEADS, TRANSFORMER_NUM_LAYERS, TRANSFORMER_FF_DIM, SEQUENCE_LENGTH

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQUENCE_LENGTH):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=1, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, TRANSFORMER_EMBED_DIM)
        self.pos_encoder = PositionalEncoding(TRANSFORMER_EMBED_DIM, max_len=SEQUENCE_LENGTH)
        encoder_layer = nn.TransformerEncoderLayer(d_model=TRANSFORMER_EMBED_DIM,
                                                   nhead=TRANSFORMER_NUM_HEADS,
                                                   dim_feedforward=TRANSFORMER_FF_DIM,
                                                   dropout=dropout,
                                                   batch_first=True)  # using batch_first for clarity
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_NUM_LAYERS)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: (batch_size, seq_len)
        src = src.unsqueeze(-1)  # (batch_size, seq_len, 1)
        src = self.input_proj(src)  # (batch_size, seq_len, embed_dim)
        src = self.pos_encoder(src)
        # Now, using batch_first transformer so we don't need to transpose
        output = self.transformer_encoder(src)  # (batch_size, seq_len, embed_dim)
        output = output.mean(dim=1)  # (batch_size, embed_dim)
        output = self.dropout(output)
        return output

if __name__ == '__main__':
    model = TransformerEncoder()
    dummy_input = torch.rand(8, SEQUENCE_LENGTH)
    encoded = model(dummy_input)
    print("Encoded shape:", encoded.shape)  # Expected: (8, TRANSFORMER_EMBED_DIM)
