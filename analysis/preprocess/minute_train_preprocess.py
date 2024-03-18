import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

def train_preprocess(args, df):
        df = df[df['minute']!=df['minute'].min()].set_index('minute')
        df = df.loc[df.index.min()+pd.Timedelta("1min"):df.index.max()-pd.Timedelta("1min")]
        ob = df[['time_floor','orderbook_pos','best_qty_ratio']].drop_duplicates()
        tr = df[['time_floor','trade_price_pos','maker_ratio','price_volume_ratio']]
        minute = df.reset_index()[['minute','minute_volume_change','minute_price_change']].drop_duplicates().set_index('minute')

        ob_minute = np.array(ob.groupby('minute')[['orderbook_pos','best_qty_ratio']].apply(lambda x: x.transpose().values.tolist()).tolist())
        tr_minute = np.array(tr.groupby('minute')[['maker_ratio','price_volume_ratio']].apply(lambda x: x.transpose().values.tolist()).tolist())
        minute = np.expand_dims(minute.to_numpy(), axis=2)
        
        data = []
        for idx in tqdm(range(0, minute.shape[0]-args.data_len-args.pred_len, args.data_hop)):
            minute_data = minute[idx:idx+args.data_len+args.pred_len].transpose(1,0,2)
            data.append({'ob':ob_minute[idx:idx+args.data_len].transpose(0,2,1).tolist(),
                         'tr':tr_minute[idx:idx+args.data_len].transpose(0,2,1).tolist(),
                         'volume':minute_data[0][:args.data_len].tolist(),
                         'tgt':minute_data[1].tolist()})
        
        return data


def main(args):
    files = os.listdir(args.data_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    for file in tqdm(files):
        df = pd.read_csv(os.path.join(args.data_dir, file))
        data = train_preprocess(args, df)

        with open(os.path.join(args.save_dir, f'train_{file.split(".csv")[0]}.json'), 'w') as f:
            json.dump(data,f)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/combined/minute')
    parser.add_argument('--save_dir', type=str, default='../data/train/minute')
    parser.add_argument('--data_len', type=int, default=20)
    parser.add_argument('--data_hop', type=int, default=5)
    parser.add_argument('--pred_len', type=int, default=1)
    args = parser.parse_args()
    main(args)