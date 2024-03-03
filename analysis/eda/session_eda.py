import pandas as pd
from argparse import ArgumentParser

def main(args):
    ob = pd.read_csv(args.orderbok_path)
    
    ob = ob[['best_bid_price', 'best_bid_qty', 'best_ask_price', 'best_ask_qty', 'transaction_time']]    
    ob['transaction_time'] = pd.to_datetime(ob['transaction_time'], unit='ms')
    ob['mid_price'] = (ob['best_bid_price'] + ob['best_ask_price']) / 2
    
    ob['mid_price_change'] = (ob['mid_price'].shift(1))!=ob['mid_price']
    ob['session'] = ob['mid_price_change'].cumsum()-1
    ob.drop(['mid_price_change'], axis=1, inplace=True)
    
    # first_time_df = ob.groupby('session')[['transaction_time']].first().rename(columns={'transaction_time': 'first_time'})
    # last_time_df = ob.groupby('session')[['transaction_time']].last().rename(columns={'transaction_time': 'last_time'})
    # session_time_df = first_time_df.merge(last_time_df, on=['session'], how='left')
    # session_time_df['session_len'] = (session_time_df['last_time'] - session_time_df['first_time']).dt.total_seconds()
    
    # ob = ob.merge(session_time_df, on=['session'], how='left')
    # del first_time_df, last_time_df, session_time_df
    
    print(ob.describe())    
    print(ob[ob['session']==5])
    
    
    tr = pd.read_csv(args.trade_path)
    
    tr = tr[['price', 'qty', 'time', 'is_buyer_maker']]
    tr['time'] = pd.to_datetime(tr['time'], unit='ms')
    
    tr = pd.concat([tr, ob[['transaction_time', 'session']].rename(columns={'transaction_time': 'time'})])
    tr.sort_values(by=['time', 'session'], inplace=True)

    tr['session'] = tr['session'].fillna(method='bfill')
    tr = tr[~tr['price'].isnull()]
    
    tr = tr.groupby(['session', 'price', 'is_buyer_maker'])[['qty']].sum().reset_index()
    print(tr[tr['session']==5])
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--orderbook_path', type=str, default='/home/jay/UpbitTrade/analysis/data/orderbook/BTCUSDT-bookTicker-2024-01-01.csv')
    parser.add_argument('--trade_path', type=str, default='/home/jay/UpbitTrade/analysis/data/trade/BTCUSDT-trades-2024-01-01.csv')
    args = parser.parse_args()
    main(args)