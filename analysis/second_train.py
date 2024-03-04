import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from model.vanilla import TransformerVanilla

def main(args):
    with open(os.path.join(args.data_dir, 'src.npy'), 'rb') as f:
        src = np.load(f)
    with open(os.path.join(args.data_dir, 'tgt.npy'), 'rb') as f:
        tgt = np.load(f)
    
    print(f'src shape: {src.shape}')
    print(f'tgt shape: {tgt.shape}')
    
    model = TransformerVanilla(model_dim=64, 
                               n_head=2, 
                               num_encoder_layers=2, 
                               num_decoder_layers=2,
                               src_dim=src.shape[-1], 
                               tgt_dim=tgt.shape[-1])
    num_param = 0
    for _, param in model.named_parameters():
        num_param += param.view(-1).shape[0]
    print(f'model param size: {num_param}')
    
    # out = model(src, tgt)
    # print(out.shape)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/jay/UpbitTrade/analysis/data/train')
    parser.add_argument('--data_len', type=int, default=120)
    parser.add_argument('--pred_gap', type=int, default=60)
    parser.add_argument('--data_hop', type=int, default=20)
    parser.add_argument('--mid_price_change_divisor', type=int, default=200)
    args = parser.parse_args()
    main(args)