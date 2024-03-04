import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TEncoderVanilla(nn.Module):
    def __init__(self, model_dim, n_head, num_layers, src_dim, tgt_dim, src_len, tgt_len):
        super().__init__()
        self.src_proj = nn.Linear(src_dim, model_dim)
        self.encoder_layer = TransformerEncoderLayer(d_model=model_dim,
                                                     dim_feedforward=model_dim*4,
                                                     nhead=n_head,
                                                     batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer,
                                          num_layers = num_layers)
        self.out_proj1 = nn.Linear(model_dim, tgt_dim)
        self.out_proj2 = nn.Linear(src_len, tgt_len)
    
    def forward(self, src, src_mask=None):
        if type(src) != torch.Tensor:
            src = torch.tensor(src)
        src = self.src_proj(src)
        out = self.encoder(src, src_mask)
        out = self.out_proj1(out).view(out.shape[0],-1)
        return self.out_proj2(out)