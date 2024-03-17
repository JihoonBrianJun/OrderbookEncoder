import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

def preprocess_orderbook(args, df):
        df = df[['best_bid_price', 'best_bid_qty', 'best_ask_price', 'best_ask_qty', 'transaction_time']]    
        df['transaction_time'] = pd.to_datetime(df['transaction_time'], unit='ms')
        df = df.sort_values(by=['transaction_time'])
        df['transaction_time_floor'] = df['transaction_time'].dt.floor(freq=f'{args.data_freq}S')
        
        df['mid_price'] = (df['best_bid_price'] + df['best_ask_price']) / 2
        df['best_qty_ratio'] = df['best_ask_qty'] / (df['best_ask_qty'] + df['best_bid_qty'])
        
        return df.groupby('transaction_time_floor')[['mid_price', 'best_qty_ratio']].last().reset_index().rename(columns={'transaction_time_floor': 'time_floor'})    


def preprocess_trade(args, df):
        df = df[['price', 'qty', 'time', 'is_buyer_maker']]
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = df.sort_values(by=['time'])
        df['time_floor'] = df['time'].dt.floor(freq=f'{args.data_freq}S')
        df['minute'] = df['time'].dt.floor(freq='min')
        
        df.sort_values(by=['time_floor'], inplace=True)
        time_interval_volume = pd.DataFrame(df.groupby(['time_floor'])['qty'].sum()).reset_index().rename(
            columns={'qty': 'time_interval_volume'})
        
        df.sort_values(by=['time'], inplace=True)
        open = pd.DataFrame(df.groupby(['minute'])['price'].first()).reset_index().rename(
            columns={'price': 'minute_open_price'})
        close = pd.DataFrame(df.groupby(['minute'])['price'].last()).reset_index().rename(
            columns={'price': 'minute_close_price'})
        
        volume = pd.DataFrame(df.groupby(['minute'])['qty'].sum()).reset_index().rename(
            columns={'qty': 'minute_volume'})
        volume['minute_volume_log'] = volume['minute_volume'].apply(lambda x: np.log(x))
        volume['minute_volume_change'] = volume['minute_volume_log'].diff()
        volume.drop(['minute_volume','minute_volume_log'], axis=1, inplace=True)
        
        df_minute = open.merge(close,on=['minute'],how='left')
        df_minute = df_minute.merge(volume,on=['minute'],how='left')
        df_minute['minute_price_range'] = (df_minute['minute_close_price']-df_minute['minute_open_price']).abs()
        df_minute['minute_price_change'] = df_minute['minute_close_price'].diff() / df_minute['minute_close_price'].shift(1) * 100
        
        df = df.merge(df_minute,on=['minute'],how='left')
        df['trade_price_pos'] = ((df['price']-df['minute_open_price']) / df['minute_price_range']).clip(-args.clip_range, args.clip_range)
        assert 20*args.clip_range % (args.price_interval_num-1) == 0 and args.price_interval_num % 10 == 1, f"Set price_interval_num to a proper value."
        round_divisor = 20*args.clip_range/(args.price_interval_num-1)
        df['trade_price_pos'] = ((df['trade_price_pos']/round_divisor).round(1))*round_divisor
        
        df = pd.DataFrame(df.groupby(['time_floor','trade_price_pos','is_buyer_maker'])['qty'].sum().unstack()).reset_index().fillna(0)
        df['price_volume'] = df[True]+df[False]
        df['maker_ratio'] = df[True] / df['price_volume']
        df.drop([True,False], axis=1, inplace=True)
        
        df = df.merge(time_interval_volume, on=['time_floor'], how='left')
        df['price_volume_ratio'] = df['price_volume'] / df['time_interval_volume']
        df.drop(['price_volume','time_interval_volume'],axis=1,inplace=True)
        
        return df, df_minute


def preprocess_combine(args, ob_df, tr_df, minute_df, date=None):
    ob_df['time_floor'] = pd.to_datetime(ob_df['time_floor'])
    ob_df['minute'] = ob_df['time_floor'].dt.floor(freq='min')
    minute_df['minute'] = pd.to_datetime(minute_df['minute'])
    ob_df = ob_df.merge(minute_df, on=['minute'], how='left')
    
    ob_df.sort_values(by=['time_floor'],inplace=True)
    ob_df['orderbook_pos'] = ((ob_df['mid_price']-ob_df['minute_open_price']) / ob_df['minute_price_range']).clip(-args.clip_range,args.clip_range)
    ob_df.drop(['mid_price','minute_open_price','minute_price_range'],axis=1,inplace=True)
    
    tr_df['time_floor'] = pd.to_datetime(tr_df['time_floor'])
    if date is None:
        all_time_floor = pd.DataFrame(pd.date_range(
            start=min(ob_df['time_floor'].min(), tr_df['time_floor'].min()), end=max(ob_df['time_floor'].max(), tr_df['time_floor'].max()),
            freq=f"{args.data_freq}s", inclusive="left"))
    else:
        all_time_floor = pd.DataFrame(pd.date_range(
            start=date, end=date+pd.Timedelta("1d"),
            freq=f"{args.data_freq}s", inclusive="left"))
        
    all = all_time_floor.rename(columns={0:'time_floor'}).merge(
        pd.DataFrame({'trade_price_pos': [i/10 for i in range(int(-10*args.clip_range),int(10*args.clip_range+1),
                                                              int(20*args.clip_range/(args.price_interval_num-1)))]}),
        how='cross'
    )

    tr_df = tr_df.merge(all,on=['time_floor','trade_price_pos'],how='outer')
    tr_df.sort_values(by=['time_floor','trade_price_pos'],inplace=True)

    agg_df = tr_df.merge(ob_df,on=['time_floor'],how='left')
    agg_df.sort_values(by=['time_floor','trade_price_pos'],inplace=True)
    agg_df.fillna(0,inplace=True)
    agg_df['minute'] = agg_df['time_floor'].dt.floor(freq='min')
    # assert agg_df.shape[0] == 24*60*(60/args.data_freq)*args.price_interval_num, f"Shape of agg is {agg_df.shape}, but it should be {24*60*(60/args.data_freq)*args.price_interval_num}."
    
    return agg_df
        

def main(args):
    orderbook_paths = [os.path.join(args.orderbook_dir, path) for path in os.listdir(args.orderbook_dir)]
    for orderbook_path in tqdm(orderbook_paths):
        ob = pd.read_csv(orderbook_path)
        ob = preprocess_orderbook(args, ob)
               
        date = '-'.join(orderbook_path.split('-')[-3:]).split('.csv')[0]
        date_path = os.path.join(args.intermediate_dir, date)
        if not os.path.exists(date_path):
            os.makedirs(date_path)
        ob.to_csv(os.path.join(date_path, 'orderbook.csv'), index=False)
    del ob
    
    
    trade_paths = [os.path.join(args.trade_dir, path) for path in os.listdir(args.trade_dir)]
    for trade_path in tqdm(trade_paths):
        tr = pd.read_csv(trade_path)
        tr, tr_minute = preprocess_trade(args, tr)
        
        date = '-'.join(trade_path.split('-')[-3:]).split('.csv')[0]
        date_path = os.path.join(args.intermediate_dir, date)
        if not os.path.exists(date_path):
            os.makedirs(date_path)
        tr.to_csv(os.path.join(date_path, 'trade.csv'), index=False)
        tr_minute.drop(['minute_close_price'],axis=1).to_csv(os.path.join(date_path, 'minute.csv'), index=False)
    del tr,tr_minute
    
    
    date_paths = [os.path.join(args.intermediate_dir, path) for path in os.listdir(args.intermediate_dir)]
    for date_path in tqdm(date_paths):
        date = pd.to_datetime(date_path.split(args.intermediate_dir+'/')[1])
        ob = pd.read_csv(os.path.join(date_path, 'orderbook.csv'))
        tr = pd.read_csv(os.path.join(date_path, 'trade.csv'))
        minute = pd.read_csv(os.path.join(date_path, 'minute.csv'))
        
        agg = preprocess_combine(args, ob, tr, minute, date)
        
        if not os.path.exists(args.final_save_dir):
            os.makedirs(args.final_save_dir)        
        save_path = os.path.join(args.final_save_dir, f"{str(date).split(' ')[0]}.csv")
        agg.to_csv(os.path.join(save_path), index=False)
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--orderbook_dir', type=str, default='/home/jay/UpbitTrade/analysis/data/orderbook')
    parser.add_argument('--trade_dir', type=str, default='/home/jay/UpbitTrade/analysis/data/trade')
    parser.add_argument('--intermediate_dir', type=str, default='/home/jay/UpbitTrade/analysis/data/processed')
    parser.add_argument('--final_save_dir', type=str, default='/home/jay/UpbitTrade/analysis/data/combined/minute')
    parser.add_argument('--data_freq', type=int, default=5)
    parser.add_argument('--clip_range', type=int, default=2)
    parser.add_argument('--price_interval_num', type=int, default=21)
    args = parser.parse_args()
    main(args)