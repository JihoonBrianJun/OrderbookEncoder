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

from model.minute import OrderbookTrade2Predictor, OrderbookTrade2Classifier
from preprocess.minute_preprocess_all import preprocess_orderbook, preprocess_trade, preprocess_combine
from preprocess.minute_train_preprocess import train_preprocess
from utils.train_utils import process_instance
from utils.test_utils import test_predictor, test_classifier


def main(args):
    with open(args.market_codes_path, 'r') as f:
        market_codes = json.load(f)
    
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    if args.model_type in ['predictor', 'hybrid']:   
        model = OrderbookTrade2Predictor(model_dim=args.model_dim,
                                         n_head=args.n_head,
                                         num_layers=args.num_layers,
                                         ob_feature_dim=int(2*(60/args.data_freq)),
                                         tr_feature_dim=int(2*(60/args.data_freq)*args.price_interval_num),
                                         volume_feature_dim=1,
                                         tgt_feature_dim=1,
                                         data_len=args.data_len,
                                         pred_len=args.pred_len,
                                         ob_importance=args.ob_importance,
                                         tr_importance=args.tr_importance).to(device)

    elif args.model_type == 'classifier':
        model = OrderbookTrade2Classifier(result_dim=args.result_dim,
                                          model_dim=args.model_dim,
                                          n_head=args.n_head,
                                          num_layers=args.num_layers,
                                          ob_feature_dim=int(2*(60/args.data_freq)),
                                          tr_feature_dim=int(2*(60/args.data_freq)*args.price_interval_num),
                                          volume_feature_dim=1,
                                          tgt_feature_dim=1,
                                          data_len=args.data_len,
                                          pred_len=args.pred_len,
                                          ob_importance=args.ob_importance,
                                          tr_importance=args.tr_importance).to(device)
        
    model.load_state_dict(torch.load(f'{args.model_ckpt_path}_{args.model_type}_{args.pred_len}.pt', map_location=device))

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
                    process_instance(ins, ins_idx, data_len_dict, feature_dim_dict)
                print(f'# of instances: {len(data)}')
                
                bs = min(len(data),args.bs)
                dataloader = DataLoader(data, batch_size=bs, shuffle=True)
                
                print(f'Loop {loop_idx} Code {market_code} Eval Result:\n')
                
                if args.model_type == 'predictor':
                    loss_function = nn.MSELoss()
                    test_predictor(model=model,
                                   loss_function=loss_function,
                                   dataloader=dataloader,
                                   test_bs=bs,
                                   data_len=args.data_len,
                                   pred_len=args.pred_len,
                                   tgt_amplifier=args.tgt_amplifier,
                                   tgt_clip_value=args.tgt_clip_value,
                                   value_threshold=args.value_threshold,
                                   strong_threshold=args.strong_threshold,
                                   device=device,
                                   save_dir=None,
                                   save_ckpt=False)
                    
                elif args.model_type == 'classifier':
                    loss_function = nn.CrossEntropyLoss()
                    test_classifier(result_dim=args.result_dim,
                                    model=model,
                                    loss_function=loss_function,
                                    dataloader=dataloader,
                                    test_bs=bs,
                                    data_len=args.data_len,
                                    pred_len=args.pred_len,
                                    tgt_amplifier=args.tgt_amplifier,
                                    tgt_clip_value=args.tgt_clip_value,
                                    value_threshold=args.value_threshold,
                                    strong_threshold=args.strong_threshold,
                                    device=device,
                                    save_dir=None,
                                    save_ckpt=False)

                elif args.model_type == 'hybrid':
                    loss_function1 = nn.MSELoss()
                    loss_function2 = nn.CrossEntropyLoss()
                    test_classifier(result_dim=args.result_dim,
                                    model=model,
                                    loss_function1=loss_function1,
                                    loss_function2=loss_function2,
                                    loss_weight=args.hybrid_loss_weight,
                                    dataloader=dataloader,
                                    test_bs=bs,
                                    data_len=args.data_len,
                                    pred_len=args.pred_len,
                                    tgt_amplifier=args.tgt_amplifier,
                                    tgt_clip_value=args.tgt_clip_value,
                                    value_threshold=args.value_threshold,
                                    strong_threshold=args.strong_threshold,
                                    device=device,
                                    save_dir=None,
                                    save_ckpt=False)
            
            except:
                print(f'Loop {loop_idx} Code {market_code} Evaluation failed!')
                pass
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--market_codes_path', type=str, default='./result/top_market_codes.json')
    parser.add_argument('--model_ckpt_path', type=str, default='ckpt/vanilla_minute')
    parser.add_argument('--loop_rep', type=int, default=3)
    parser.add_argument('--loop_len', type=int, default=10800)
    parser.add_argument('--data_len', type=int, default=20)
    parser.add_argument('--data_hop', type=int, default=5)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--data_freq', type=int, default=5)
    parser.add_argument('--clip_range', type=int, default=2)
    parser.add_argument('--price_interval_num', type=int, default=21)
    parser.add_argument('--model_type', type=str, default='predictor', choices=['predictor', 'classifier', 'hybrid'])
    parser.add_argument('--result_dim', type=int, default=3)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--tgt_amplifier', type=float, default=10)
    parser.add_argument('--tgt_clip_value', type=float, default=1)
    parser.add_argument('--value_threshold', type=float, default=0.5)
    parser.add_argument('--strong_threshold', type=float, default=0.5)
    parser.add_argument('--ob_importance', type=float, default=0.4)
    parser.add_argument('--tr_importance', type=float, default=0.4)
    parser.add_argument('--hybrid_loss_weight', type=float, default=0.5)
    args = parser.parse_args()
    main(args)