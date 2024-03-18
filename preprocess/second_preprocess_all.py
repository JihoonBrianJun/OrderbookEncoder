import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

def main(args):
    orderbook_paths = [os.path.join(args.orderbook_dir, path) for path in os.listdir(args.orderbook_dir)]
    for orderbook_path in tqdm(orderbook_paths):
        ob = pd.read_csv(orderbook_path)
        
        ob = ob[['best_bid_price', 'best_bid_qty', 'best_ask_price', 'best_ask_qty', 'transaction_time']]    
        ob['transaction_time'] = pd.to_datetime(ob['transaction_time'], unit='ms')
        ob['transaction_sec'] = ob['transaction_time'].dt.floor(freq='S')
        
        price_gap = np.round((ob['best_ask_price'] - ob['best_bid_price']).mode().item(), 6)
        base_price = ob['best_bid_price'].iloc[0].item() + (price_gap / 2)
        
        ob['mid_price'] = ((((ob['best_bid_price'] + ob['best_ask_price']) / 2) - base_price) / price_gap).round(6)
        ob['qty_ratio'] = ob['best_ask_qty'] / ob['best_bid_qty']
        
        ob = ob.groupby('transaction_sec')[['mid_price', 'qty_ratio']].last().reset_index().rename(columns={'transaction_sec': 'sec'})
        
        date = '-'.join(orderbook_path.split('-')[-3:]).split('.csv')[0]
        date_path = os.path.join(args.intermediate_dir, date)
        if not os.path.exists(date_path):
            os.makedirs(date_path)
        ob.to_csv(os.path.join(date_path, 'orderbook.csv'), index=False)
    del ob
    
    
    trade_paths = [os.path.join(args.trade_dir, path) for path in os.listdir(args.trade_dir)]
    for trade_path in tqdm(trade_paths):
        tr = pd.read_csv(trade_path)
        
        tr = tr[['price', 'qty', 'time', 'is_buyer_maker']]
        tr['time'] = pd.to_datetime(tr['time'], unit='ms')
        tr['sec'] = tr['time'].dt.floor(freq='S')
        
        tr = tr.groupby(['sec', 'is_buyer_maker'])['qty'].sum().unstack()
        tr = pd.DataFrame(tr[True] / tr[False]).rename(columns={0:'maker_ratio'}).reset_index()
        
        date = '-'.join(trade_path.split('-')[-3:]).split('.csv')[0]
        date_path = os.path.join(args.intermediate_dir, date)
        if not os.path.exists(date_path):
            os.makedirs(date_path)
        tr.to_csv(os.path.join(date_path, 'trade.csv'), index=False)
    del tr
    
    
    date_paths = [os.path.join(args.intermediate_dir, path) for path in os.listdir(args.intermediate_dir)]
    for date_path in tqdm(date_paths):
        ob = pd.read_csv(os.path.join(date_path, 'orderbook.csv'))
        tr = pd.read_csv(os.path.join(date_path, 'trade.csv'))
        
        agg= ob.merge(tr, on=['sec'], how='left')
        del ob, tr
        agg['qty_ratio'] = agg['qty_ratio'].apply(lambda x: np.log(x))
        agg['maker_ratio'] = agg['maker_ratio'].apply(lambda x: np.log(x)).fillna(method='ffill')
        agg['mid_price_change'] = agg['mid_price'].diff()

        if not os.path.exists(args.final_save_dir):
            os.makedirs(args.final_save_dir)   
        save_path = os.path.join(args.final_save_dir, f"{date_path.split(args.intermediate_dir+'/')[1]}.csv")
        agg.to_csv(os.path.join(save_path), index=False)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--orderbook_dir', type=str, default='../data/orderbook')
    parser.add_argument('--trade_dir', type=str, default='../data/trade')
    parser.add_argument('--intermediate_dir', type=str, default='../data/processed')
    parser.add_argument('--final_save_dir', type=str, default='../data/combined/second')
    args = parser.parse_args()
    main(args)