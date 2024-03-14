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
        ob['transaction_time_floor'] = ob['transaction_time'].dt.floor(freq=f'{args.data_freq}S')
        
        ob['mid_price'] = (ob['best_bid_price'] + ob['best_ask_price']) / 2
        ob['best_qty_ratio'] = ob['best_ask_qty'] / (ob['best_ask_qty'] + ob['best_bid_qty'])
        
        ob = ob.groupby('transaction_time_floor')[['mid_price', 'best_qty_ratio']].last().reset_index().rename(columns={'transaction_time_floor': 'time_floor'})
                
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
        tr['time_floor'] = tr['time'].dt.floor(freq=f'{args.data_freq}S')
        tr['minute'] = tr['time'].dt.floor(freq='min')
        
        tr.sort_values(by=['time_floor'], inplace=True)
        tr_interval_volume = pd.DataFrame(tr.groupby(['time_floor'])['qty'].sum()).reset_index().rename(
            columns={'qty': 'interval_volume'})
        
        tr.sort_values(by=['time'], inplace=True)
        tr_open = pd.DataFrame(tr.groupby(['minute'])['price'].first()).reset_index().rename(
            columns={'price': 'minute_open_price'})
        tr_close = pd.DataFrame(tr.groupby(['minute'])['price'].last()).reset_index().rename(
            columns={'price': 'minute_close_price'})
        
        tr_volume = pd.DataFrame(tr.groupby(['minute'])['qty'].sum()).reset_index().rename(
            columns={'qty': 'minute_volume'})
        tr_volume['minute_volume_log'] = tr_volume['minute_volume'].apply(lambda x: np.log(x))
        tr_volume['minute_volume_change'] = tr_volume['minute_volume_log'].diff()
        tr_volume.drop(['minute_volume','minute_volume_log'], axis=1, inplace=True)
        
        tr_minute = tr_open.merge(tr_close,on=['minute'],how='left')
        tr_minute = tr_minute.merge(tr_volume,on=['minute'],how='left')
        del tr_open,tr_close,tr_volume
        tr_minute['minute_price_range'] = (tr_minute['minute_close_price']-tr_minute['minute_open_price']).abs()
        tr_minute['minute_price_change'] = tr_minute['minute_close_price'].diff() / tr_minute['minute_close_price'].shift(1) * 100
        
        tr = tr.merge(tr_minute,on=['minute'],how='left')
        tr['trade_price_pos'] = ((tr['price']-tr['minute_open_price']) / tr['minute_price_range']).clip(-args.clip_range, args.clip_range)
        assert 20*args.clip_range % (args.price_interval_num-1) == 0 and args.price_interval_num % 10 == 1, f"Set price_interval_num to a proper value."
        round_divisor = 20*args.clip_range/(args.price_interval_num-1)
        tr['trade_price_pos'] = ((tr['trade_price_pos']/round_divisor).round(1))*round_divisor
        
        tr = pd.DataFrame(tr.groupby(['time_floor','trade_price_pos','is_buyer_maker'])['qty'].sum().unstack()).reset_index().fillna(0)
        tr['price_volume'] = tr[True]+tr[False]
        tr['maker_ratio'] = tr[True] / tr['price_volume']
        tr.drop([True,False], axis=1, inplace=True)
        
        tr = tr.merge(tr_interval_volume, on=['time_floor'], how='left')
        del tr_interval_volume
        tr['price_volume_ratio'] = tr['price_volume'] / tr['interval_volume']
        tr.drop(['price_volume','interval_volume'],axis=1,inplace=True)
        
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

        ob['time_floor'] = pd.to_datetime(ob['time_floor'])
        ob['minute'] = ob['time_floor'].dt.floor(freq='min')
        minute['minute'] = pd.to_datetime(minute['minute'])
        ob = ob.merge(minute, on=['minute'], how='left')
        del minute
        
        ob.sort_values(by=['time_floor'],inplace=True)
        ob['orderbook_pos'] = ((ob['mid_price']-ob['minute_open_price']) / ob['minute_price_range']).clip(-args.clip_range,args.clip_range)
        ob.drop(['mid_price','minute_open_price','minute_price_range'],axis=1,inplace=True)
        
        all = pd.DataFrame(pd.date_range(
            start=date, end=date+pd.Timedelta("1d"),
            freq=f"{args.data_freq}s", inclusive="left")).rename(columns={0:'time_floor'}).merge(
            pd.DataFrame({'trade_price_pos': [i/10 for i in range(int(-10*args.clip_range),int(10*args.clip_range+1),
                                                                  int(20*args.clip_range/(args.price_interval_num-1)))]}),
            how='cross'
        )

        tr['time_floor'] = pd.to_datetime(tr['time_floor'])
        tr = tr.merge(all,on=['time_floor','trade_price_pos'],how='outer')
        del all
        tr.sort_values(by=['time_floor','trade_price_pos'],inplace=True)

        agg = tr.merge(ob,on=['time_floor'],how='left')
        agg.sort_values(by=['time_floor','trade_price_pos'],inplace=True)
        agg.fillna(0,inplace=True)
        agg['minute'] = agg['time_floor'].dt.floor(freq='min')
        assert agg.shape[0] == 24*60*(60/args.data_freq)*args.price_interval_num, f"Shape of agg is {agg.shape}, but it should be {24*60*(60/args.data_freq)*args.price_interval_num}."

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