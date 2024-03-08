import os
import json
import requests
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from analysis.model.obtr2pr import OrderbookTrade2Price, AnomalyDetector

def main(args):
    with open(args.market_codes_path, 'r') as f:
        market_codes = json.load(f)
    
    orderbook_data = []
    trade_data = []
    for idx in tqdm(range(args.loop_len)):
        for market_code in market_codes:
            try:
                orderbook_url = f"https://api.upbit.com/v1/orderbook?markets={market_code}"
                headers = {"accept": "application/json"}
                response = requests.get(orderbook_url, headers=headers).json()[0]
                
                best_orderbook = response['orderbook_units'][0]
                best_orderbook.update({'timestamp': response['timestamp'], 'code': market_code})
                orderbook_data.append(best_orderbook)

                if idx % 10 == 0:
                    trade_url = "https://api.upbit.com/v1/trades/ticks?count=500"
                    headers = {"accept": "application/json"}
                    params = {"market": market_code}
                    response = requests.get(trade_url, headers=headers, params=params).json()
                    trade_data.extend(response)
            except:
                pass
        time.sleep(0.4)
        

    
    all_ob = pd.DataFrame(orderbook_data)
    all_ob['sec'] = pd.to_datetime(all_ob['timestamp'], unit='ms').dt.floor(freq='S')
    all_ob.drop(['timestamp'], axis=1, inplace=True)
    
    all_tr = pd.DataFrame(trade_data)
    
    device = torch.device("cpu")
    
    for market_code in market_codes:
        try:
            ob = all_ob[all_ob['code']==market_code]
            ob.drop(['code'], axis=1, inplace=True)
            ob.sort_values(by=['sec'], inplace=True)
            
            price_gap = np.round((ob['ask_price'] - ob['bid_price']).mode().item(), 6)
            base_price = ob['bid_price'].iloc[0].item() + (price_gap / 2)

            ob['mid_price'] = ((((ob['bid_price'] + ob['ask_price']) / 2) - base_price) / price_gap).round(6)
            ob['qty_ratio'] = ob['ask_size'] / ob['bid_size']

            ob = ob.groupby('sec')[['mid_price', 'qty_ratio']].last().reset_index().drop_duplicates()
            
            tr = all_tr[all_tr['market']==market_code].drop_duplicates()
            tr = tr[['trade_date_utc', 'trade_time_utc', 'trade_price', 'trade_volume', 'ask_bid']]
            tr['sec'] = pd.to_datetime(tr['trade_date_utc'] + ' ' + tr['trade_time_utc'])
            tr.drop(['trade_date_utc', 'trade_time_utc'], axis=1, inplace=True)
            
            tr = tr.groupby(['sec', 'ask_bid'])['trade_volume'].sum().unstack()
            tr = pd.DataFrame(tr['ASK'] / tr['BID']).rename(columns={0:'maker_ratio'}).reset_index() 
            
            
            agg = ob.merge(tr, on=['sec'], how='left')
            agg['qty_ratio'] = agg['qty_ratio'].apply(lambda x: np.log(x))
            agg['maker_ratio'] = agg['maker_ratio'].apply(lambda x: np.log(x)).fillna(method='ffill')
            agg['mid_price_change'] = agg['mid_price'].diff()            
            
            agg = agg.dropna()
            agg['gap_mid_price_change'] = agg['mid_price'].diff(periods=args.pred_hop) / agg['mid_price'].diff(periods=args.pred_hop).std()
            # src = np.stack([agg.iloc[i:i+args.data_len][['qty_ratio', 'maker_ratio']].to_numpy()
            #                 for i in range(0, agg.shape[0]-args.data_len-args.pred_len, args.data_hop)], axis=0)
            # tgt = np.stack([agg.iloc[i:i+args.pred_len]['gap_mid_price_change'].to_numpy()
            #                 for i in range(args.data_len, agg.shape[0]-args.pred_len, args.data_hop)], axis=0)
            # tgt = np.expand_dims(tgt[:,args.pred_hop-1::args.pred_hop], axis=2)

            src = np.stack([agg.iloc[i:i+args.data_len][['qty_ratio', 'maker_ratio']].to_numpy()
                            for i in range(0, agg.shape[0]-args.data_len-2*args.pred_len, args.data_hop)], axis=0) 
            tgt = np.stack([agg.iloc[i:i+args.data_len+args.pred_len]['gap_mid_price_change'].to_numpy()
                            for i in range(args.pred_len, agg.shape[0]-args.data_len-args.pred_len, args.data_hop)], axis=0)
            tgt = np.expand_dims(tgt[:,::args.pred_hop], axis=2)

            print(f'src shape: {src.shape}')
            print(f'tgt shape: {tgt.shape}')
            
            # code_path = os.path.join('real_data', market_code)
            # if not os.path.exists(code_path):
            #     os.makedirs(code_path)
            # with open(os.path.join(code_path, 'src.npy'), 'wb') as f:
            #     np.save(f, src)
            # with open(os.path.join(code_path, 'tgt.npy'), 'wb') as f:
            #     np.save(f, tgt)
                        
            
            model = AnomalyDetector(model_dim=args.model_dim,
                                    n_head=args.n_head,
                                    num_layers=args.num_layers,
                                    src_dim=src.shape[-1],
                                    tgt_dim=tgt.shape[-1],
                                    src_len=src.shape[-2],
                                    tgt_len=tgt.shape[-2],
                                    result_dim=args.result_dim).to(device)
            model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device))

            src = torch.tensor(src).to(torch.float32).to(device)
            tgt = torch.clamp(torch.tensor(tgt),
                              min=-args.tgt_clip_value,
                              max=args.tgt_clip_value).to(torch.float32).to(device)
            
            out = model(src, tgt[:,:-1,:])
            label = tgt[:,1:,:].squeeze(dim=2)
            label = 1+(label>=args.value_threshold).to(torch.long)-(label<=-args.value_threshold).to(torch.long)

            rec_tgt = (label[:,-1]!=1).to(torch.long).sum().item()
            rec_correct = ((label[:,-1]!=1).to(torch.long) * (torch.argmax(out[:,-1,:],dim=1)==label[:,-1]).to(torch.long)).sum().item()

            prec_tgt = (torch.argmax(out[:,-1,:],dim=1)!=1).to(torch.long).sum().item()
            prec_correct = ((torch.argmax(out[:,-1,:],dim=1)!=1).to(torch.long) * (torch.argmax(out[:,-1,:],dim=1)==label[:,-1]).to(torch.long)).sum().item()

            strong_prec_tgt = ((torch.max(out[:,-1,:],dim=1).values>=args.strong_threshold).to(torch.long) * (torch.argmax(out[:,-1,:],dim=1)!=1).to(torch.long)).sum().item()
            strong_prec_correct = ((torch.max(out[:,-1,:],dim=1).values>=args.strong_threshold).to(torch.long) * (torch.argmax(out[:,-1,:],dim=1)!=1).to(torch.long) * (torch.argmax(out[:,-1,:],dim=1)==label[:,-1]).to(torch.long)).sum().item()

            print(f'Out: {out[:,-1,:]}\n Label: {label[:,-1]}')
            print(f'Recall: {rec_correct} out of {rec_tgt}')
            print(f'Precision: {prec_correct} out of {prec_tgt}')
            print(f'Precision (Strong): {strong_prec_correct} out of {strong_prec_tgt}')
        
        except:
            pass
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--market_codes_path', type=str, default='./result/top_market_codes.json')
    parser.add_argument('--model_ckpt_path', type=str, default='analysis/vanilla.pt')
    parser.add_argument('--loop_len', type=int, default=600)
    parser.add_argument('--data_len', type=int, default=120)
    parser.add_argument('--data_hop', type=int, default=20)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--pred_hop', type=int, default=10)
    parser.add_argument('--mid_price_change_divisor', type=int, default=1)
    parser.add_argument('--result_dim', type=int, default=3)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--tgt_clip_value', type=float, default=1)
    parser.add_argument('--value_threshold', type=float, default=0.5)
    parser.add_argument('--strong_threshold', type=float, default=0.9)
    args = parser.parse_args()
    main(args)