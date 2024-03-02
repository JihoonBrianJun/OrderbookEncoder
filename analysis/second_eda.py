import requests
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser

def main(args):
    ob = pd.read_csv('data/orderbook/BTCUSDT-bookTicker-2024-01-01.csv')
    
    ob = ob[['best_bid_price', 'best_bid_qty', 'best_ask_price', 'best_ask_qty', 'transaction_time']]    
    ob['transaction_time'] = pd.to_datetime(ob['transaction_time'], unit='ms')
    ob['transaction_sec'] = ob['transaction_time'].dt.floor(freq='S')
    
    price_gap = np.round((ob['best_ask_price'] - ob['best_bid_price']).mode().item(), 6)
    base_price = ob['best_bid_price'].iloc[0].item() + (price_gap / 2)
    
    ob['mid_price'] = ((((ob['best_bid_price'] + ob['best_ask_price']) / 2) - base_price) / price_gap).round(6)
    ob['qty_ratio'] = ob['best_ask_qty'] / ob['best_bid_qty']
    
    ob = ob.groupby('transaction_sec')[['mid_price', 'qty_ratio']].last().reset_index().rename(columns={'transaction_sec': 'sec'})
    
    
    tr = pd.read_csv('data/trade/BTCUSDT-trades-2024-01-01.csv')
    
    tr = tr[['price', 'qty', 'time', 'is_buyer_maker']]
    tr['time'] = pd.to_datetime(tr['time'], unit='ms')
    tr['sec'] = tr['time'].dt.floor(freq='S')
    
    tr = tr.groupby(['sec', 'is_buyer_maker'])['qty'].sum().unstack()
    tr = pd.DataFrame(tr[True] / tr[False]).rename(columns={0:'maker_ratio'}).reset_index()
    
    agg= ob.merge(tr, on=['sec'], how='left')
    del ob, tr
    agg['qty_ratio'] = agg['qty_ratio'].apply(lambda x: np.log(x))
    agg['maker_ratio'] = agg['maker_ratio'].apply(lambda x: np.log(x)).fillna(method='ffill')
    agg['mid_price_change'] = agg['mid_price'].diff()
    
    print(agg.head(20))
    print(agg.drop(['sec','mid_price'], axis=1).corr())
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    main(args)