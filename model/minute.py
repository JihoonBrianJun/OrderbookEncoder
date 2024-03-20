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


class OrderbookTradeTransformer(nn.Module):
    def __init__(self, model_dim, n_head, num_layers, 
                 ob_feature_dim, tr_feature_dim, volume_feature_dim, tgt_feature_dim,
                 data_len, pred_len, ob_importance=0.4, tr_importance=0.4):
        super().__init__()
        ob_proj_dim = int(model_dim*ob_importance)
        tr_proj_dim = int(model_dim*tr_importance)
        
        self.ob_proj = nn.Linear(ob_feature_dim, ob_proj_dim)
        self.tr_proj = nn.Linear(tr_feature_dim, tr_proj_dim)
        self.volume_proj = nn.Linear(volume_feature_dim, model_dim-ob_proj_dim-tr_proj_dim)
        self.tgt_proj = nn.Linear(tgt_feature_dim, model_dim)
        
        self.in_pos_emb = PositionalEncoding(d_model=model_dim,
                                             max_len=data_len)
        self.out_pos_emb = PositionalEncoding(d_model=model_dim,
                                              max_len=data_len+pred_len)
        
        self.transformer = Transformer(d_model=model_dim,
                                       nhead=n_head,
                                       num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers,
                                       dim_feedforward=model_dim*4,
                                       batch_first=True)
    
    def forward(self, ob, tr, volume, tgt, src_mask=None, tgt_mask=None):
        ob = self.ob_proj(ob)
        tr = self.tr_proj(tr)
        volume = self.volume_proj(volume)
        
        src = self.in_pos_emb(torch.cat((ob,tr,volume),dim=2))
        if src_mask is not None:
            src_mask = src_mask.to(src.device)
            
        tgt = self.out_pos_emb(self.tgt_proj(tgt))
        if tgt_mask is None:
            tgt_mask = Transformer.generate_square_subsequent_mask(tgt.shape[1])
        tgt_mask = tgt_mask.to(tgt.device)
        
        return self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)


class OrderbookTrade2Predictor(nn.Module):
    def __init__(self, model_dim, n_head, num_layers, 
                 ob_feature_dim, tr_feature_dim, volume_feature_dim, tgt_feature_dim,
                 data_len, pred_len, ob_importance=0.4, tr_importance=0.4):
        super().__init__()
        self.orderbook_trade_transformer = OrderbookTradeTransformer(model_dim, n_head, num_layers,
                                                                     ob_feature_dim, tr_feature_dim, volume_feature_dim, tgt_feature_dim,
                                                                     data_len, pred_len, ob_importance, tr_importance)
        self.out_proj = nn.Linear(model_dim, 1)
    
    def forward(self, ob, tr, volume, tgt, src_mask=None, tgt_mask=None):
        out = self.orderbook_trade_transformer(ob, tr, volume, tgt, src_mask, tgt_mask)
        return self.out_proj(out).squeeze(dim=2)


class OrderbookTrade2Classifier(nn.Module):
    def __init__(self, result_dim, model_dim, n_head, num_layers, 
                 ob_feature_dim, tr_feature_dim, volume_feature_dim, tgt_feature_dim,
                 data_len, pred_len, ob_importance=0.4, tr_importance=0.4):
        super().__init__()
        self.pred_len = pred_len
        self.result_dim = result_dim
        self.orderbook_trade_transformer = OrderbookTradeTransformer(model_dim, n_head, num_layers,
                                                                     ob_feature_dim, tr_feature_dim, volume_feature_dim, tgt_feature_dim,
                                                                     data_len, pred_len, ob_importance, tr_importance)
        self.out_proj = nn.Linear(model_dim, pred_len*result_dim)
        self.softmax = nn.Softmax(dim=3)
    
    def forward(self, ob, tr, volume, tgt, src_mask=None, tgt_mask=None):
        out = self.orderbook_trade_transformer(ob, tr, volume, tgt, src_mask, tgt_mask)
        return self.softmax(self.out_proj(out).view(-1,tgt.shape[1],self.pred_len,self.result_dim)).view(-1,tgt.shape[1],self.pred_len*self.result_dim)