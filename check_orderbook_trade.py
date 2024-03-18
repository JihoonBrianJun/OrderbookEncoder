import os
import json
import requests
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.minute import OrderbookTrade2Price
from preprocess.minute_preprocess_all import preprocess_orderbook, preprocess_trade, preprocess_combine
from preprocess.minute_train_preprocess import train_preprocess


def main(args):
    with open(args.market_codes_path, 'r') as f:
        market_codes = json.load(f)
    
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = OrderbookTrade2Price(model_dim=args.model_dim,
                                 n_head=args.n_head,
                                 num_layers=args.num_layers,
                                 ob_feature_dim=int(2*(60/args.data_freq)),
                                 tr_feature_dim=int(2*(60/args.data_freq)*args.price_interval_num),
                                 volume_feature_dim=1,
                                 tgt_feature_dim=1,
                                 data_len=args.data_len,
                                 ob_importance=args.ob_importance,
                                 tr_importance=args.tr_importance).to(device)
    model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device))

    for loop_idx in tqdm(range(args.loop_rep)):
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
        
        
        all_ob = pd.DataFrame(orderbook_data).rename(columns={"timestamp": "transaction_time",
                                                              "bid_price": "best_bid_price",
                                                              "ask_price": "best_ask_price",
                                                              "bid_size": "best_bid_qty",
                                                              "ask_size": "best_ask_qty"})
        
        all_tr = pd.DataFrame(trade_data).rename(columns={"trade_price": "price",
                                                          "trade_volume": "qty"})
        all_tr["time"] = pd.to_datetime(all_tr["trade_date_utc"] + ' ' + all_tr["trade_time_utc"])
        all_tr["is_buyer_maker"] = all_tr["ask_bid"].apply(lambda x: True if x=="ASK" else False)
        
        for market_code in market_codes:
            try:
                ob = all_ob[all_ob['code']==market_code]
                print(f'Loop {loop_idx} Code {market_code} Orderbook Data Len: {ob.shape[0]}')
                ob = preprocess_orderbook(args, ob)
                
                tr = all_tr[all_tr['market']==market_code].drop_duplicates()
                print(f'Loop {loop_idx} Code {market_code} Trade Data Len: {tr.shape[0]}')
                tr, tr_minute = preprocess_trade(args, tr)        
                
                agg = preprocess_combine(args, ob, tr, tr_minute)
                data = train_preprocess(args, agg)
                
                data_len_dict, feature_dim_dict = dict(), dict()
                for ins_idx, ins in enumerate(data):
                    for key in ins.keys():
                        ins[key] = np.array(ins[key]).reshape(len(ins[key]),-1)
                        if ins_idx == 0:
                            print(f'Each {key} shape: {ins[key].shape}')
                            data_len_dict[key] = ins[key].shape[0]
                            feature_dim_dict[key] = ins[key].shape[1]
                print(f'# of instances: {len(data)}')
                
                bs = min(len(data),args.bs)
                dataloader = DataLoader(data, batch_size=bs, shuffle=True)
                loss_function = nn.MSELoss()
                
                model.eval()
                test_loss = 0
                correct = 0
                rec_correct, rec_tgt = 0,0
                strong_prec_correct, strong_prec_tgt = 0,0
                for idx, batch in enumerate(dataloader):
                    ob = batch['ob'].to(torch.float32).to(device)
                    tr = batch['tr'].to(torch.float32).to(device)
                    volume = batch['volume'].to(torch.float32).to(device)
                    tgt = torch.clamp(batch['tgt']*args.tgt_amplifier,
                                      min=-args.tgt_clip_value,
                                      max=args.tgt_clip_value).to(torch.float32).to(device)
                    
                    out = model(ob, tr, volume, tgt[:,:-1,:])
                    label = tgt[:,1:,:].squeeze(dim=2)
                    
                    loss = loss_function(out,label)
                    test_loss += loss.detach().cpu().item()
                    correct += ((out[:,-1]*label[:,-1])>0).sum().item()

                    rec_tgt += (label[:,-1]>=args.value_threshold).to(torch.long).sum().item()
                    rec_correct += ((label[:,-1]>=args.value_threshold).to(torch.long) * (out[:,-1]>0).to(torch.long)).sum().item()

                    strong_prec_tgt += (out[:,-1]>=args.strong_threshold).to(torch.long).sum().item()
                    strong_prec_correct += ((out[:,-1]>=args.strong_threshold).to(torch.long) * (label[:,-1]>0).to(torch.long)).sum().item()

                print(f'Loop {loop_idx} Code {market_code} Out: {out[:,-1]}\n Label: {label[:,-1]}')
                print(f'Loop {loop_idx} Code {market_code} Average Loss: {test_loss / (idx+1)}')
                print(f'Loop {loop_idx} Code {market_code} Correct: {correct} out of {bs*(idx+1)}')
                print(f'Loop {loop_idx} Code {market_code} Recall: {rec_correct} out of {rec_tgt}')
                print(f'Loop {loop_idx} Code {market_code} Precision (Strong): {strong_prec_correct} out of {strong_prec_tgt}')
            
            except:
                print(f'Loop {loop_idx} Code {market_code} Evaluation failed!')
                pass
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--market_codes_path', type=str, default='./result/top_market_codes.json')
    parser.add_argument('--model_ckpt_path', type=str, default='analysis/vanilla_minute.pt')
    parser.add_argument('--loop_rep', type=int, default=3)
    parser.add_argument('--loop_len', type=int, default=10800)
    parser.add_argument('--data_len', type=int, default=20)
    parser.add_argument('--data_hop', type=int, default=5)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--data_freq', type=int, default=5)
    parser.add_argument('--clip_range', type=int, default=2)
    parser.add_argument('--price_interval_num', type=int, default=21)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--tgt_amplifier', type=float, default=10)
    parser.add_argument('--tgt_clip_value', type=float, default=1)
    parser.add_argument('--value_threshold', type=float, default=0.5)
    parser.add_argument('--strong_threshold', type=float, default=0.9)
    parser.add_argument('--ob_importance', type=float, default=0.4)
    parser.add_argument('--tr_importance', type=float, default=0.4)
    args = parser.parse_args()
    main(args)