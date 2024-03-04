import torch
import torch.nn as nn
from torch.nn import Transformer

class TransformerVanilla(nn.Module):
    def __init__(self, model_dim, n_head, num_encoder_layers, num_decoder_layers, src_dim, tgt_dim):
        super().__init__()
        self.src_proj = nn.Linear(src_dim, model_dim)
        self.tgt_proj = nn.Linear(tgt_dim, model_dim)
        self.transformer = Transformer(d_model=model_dim, 
                                       nhead=n_head, 
                                       num_encoder_layers=num_encoder_layers, 
                                       num_decoder_layers=num_decoder_layers, 
                                       batch_first=True)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_proj(torch.tensor(src).float())
        tgt = self.tgt_proj(torch.tensor(tgt).float())
        out = self.transformer(src, tgt, src_mask, tgt_mask)
        return out