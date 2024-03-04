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
        self.out_proj = nn.Linear(model_dim, tgt_dim)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if type(src) != torch.Tensor:
            src = torch.tensor(src)
            tgt = torch.tensor(tgt)
        src = self.src_proj(src)
        tgt = self.tgt_proj(tgt)
        out = self.transformer(src, tgt, src_mask, tgt_mask)
        return self.out_proj(out)