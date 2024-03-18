import math
import torch
import torch.nn as nn
from torch.nn import Transformer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        return (self.dropout(x)).transpose(0,1)

class OrderbookTrade2Price(nn.Module):
    def __init__(self, model_dim, n_head, num_layers, src_dim, tgt_dim, src_len, tgt_len):
        super().__init__()
        self.src_proj = nn.Linear(src_dim, model_dim)
        self.src_pos_emb = PositionalEncoding(d_model=model_dim,
                                              max_len=src_len)
        
        self.tgt_proj = nn.Linear(tgt_dim, model_dim)
        self.tgt_pos_emb = PositionalEncoding(d_model=model_dim,
                                              max_len=tgt_len)
        
        self.transformer = Transformer(d_model=model_dim,
                                       nhead=n_head,
                                       num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers,
                                       dim_feedforward=model_dim*4,
                                       batch_first=True)
        self.out_proj = nn.Linear(model_dim, tgt_dim)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if type(src) != torch.Tensor:
            src = torch.tensor(src)
        src = self.src_proj(src)
        src = self.src_pos_emb(src)
        
        if src_mask is not None:
            src_mask = src_mask.to(src.device)
        if tgt_mask is None:
            tgt_mask = Transformer.generate_square_subsequent_mask(tgt.shape[1])
        tgt_mask = tgt_mask.to(tgt.device)
        
        tgt = self.tgt_proj(tgt)
        tgt = self.tgt_pos_emb(tgt)
        out = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.out_proj(out).view(out.shape[0],-1)


class AnomalyDetector(nn.Module):
    def __init__(self, model_dim, n_head, num_layers, src_dim, tgt_dim, src_len, tgt_len, result_dim):
        super().__init__()
        self.src_proj = nn.Linear(src_dim, model_dim)
        self.src_pos_emb = PositionalEncoding(d_model=model_dim,
                                              max_len=src_len)
        
        self.tgt_proj = nn.Linear(tgt_dim, model_dim)
        self.tgt_pos_emb = PositionalEncoding(d_model=model_dim,
                                              max_len=tgt_len)
        
        self.transformer = Transformer(d_model=model_dim,
                                       nhead=n_head,
                                       num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers,
                                       dim_feedforward=model_dim*4,
                                       batch_first=True)
        self.out_proj = nn.Linear(model_dim, result_dim)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if type(src) != torch.Tensor:
            src = torch.tensor(src)
        src = self.src_proj(src)
        src = self.src_pos_emb(src)
        
        if src_mask is not None:
            src_mask = src_mask.to(src.device)
        if tgt_mask is None:
            tgt_mask = Transformer.generate_square_subsequent_mask(tgt.shape[1])
        tgt_mask = tgt_mask.to(tgt.device)
        
        tgt = self.tgt_proj(tgt)
        tgt = self.tgt_pos_emb(tgt)
        out = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.softmax(self.out_proj(out))