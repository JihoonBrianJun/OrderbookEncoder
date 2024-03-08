import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

def main(args):
    all_src, all_tgt = None, None
    files = [os.path.join(args.data_dir, file) for file in os.listdir(args.data_dir)]
    
    for file in tqdm(files):
        df = pd.read_csv(file).dropna()
        print(df['mid_price'].diff(periods=args.pred_hop).describe())
        df['gap_mid_price_change'] = df['mid_price'].diff(periods=args.pred_hop) / args.mid_price_change_divisor
        
        # src = np.stack([df.iloc[i:i+args.data_len][['qty_ratio', 'maker_ratio']].to_numpy()
        #                 for i in range(0, df.shape[0]-args.data_len-args.pred_len, args.data_hop)], axis=0) 
        # tgt = np.stack([df.iloc[i:i+args.pred_len]['gap_mid_price_change'].to_numpy()
        #                 for i in range(args.data_len, df.shape[0]-args.pred_len, args.data_hop)], axis=0)
        # tgt = np.expand_dims(tgt[:,args.pred_hop-1::args.pred_hop], axis=2)
        
        src = np.stack([df.iloc[i:i+args.data_len][['qty_ratio', 'maker_ratio']].to_numpy()
                        for i in range(0, df.shape[0]-args.data_len-2*args.pred_len, args.data_hop)], axis=0) 
        tgt = np.stack([df.iloc[i:i+args.data_len+args.pred_len]['gap_mid_price_change'].to_numpy()
                        for i in range(args.pred_len, df.shape[0]-args.data_len-args.pred_len, args.data_hop)], axis=0)
        tgt = np.expand_dims(tgt[:,::args.pred_hop], axis=2)
        
        
        if all_src is None:
            all_src = src
            all_tgt = tgt
        else:
            all_src = np.concatenate((all_src, src), axis=0)
            all_tgt = np.concatenate((all_tgt, tgt), axis=0)


    with open(os.path.join(args.save_dir, 'src.npy'), 'wb') as f:
        np.save(f, all_src)
    with open(os.path.join(args.save_dir, 'tgt.npy'), 'wb') as f:
        np.save(f, all_tgt)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/jay/UpbitTrade/analysis/data/combined')
    parser.add_argument('--save_dir', type=str, default='/home/jay/UpbitTrade/analysis/data/train')
    parser.add_argument('--data_len', type=int, default=120)
    parser.add_argument('--data_hop', type=int, default=20)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--pred_hop', type=int, default=10)
    parser.add_argument('--mid_price_change_divisor', type=int, default=70)
    args = parser.parse_args()
    main(args)