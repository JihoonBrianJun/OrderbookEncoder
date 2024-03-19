import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model.minute import OrderbookTrade2Price


def main(args):
    data_dir = f'{args.data_dir}_{args.pred_len}'
    save_dir = f'{args.save_dir}_{args.pred_len}.pt'
    if not os.path.exists(save_dir.split('/')[0]):
        os.makedirs(save_dir.split('/')[0])
    
    data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
    train_data, test_data = [], []
    data_len_dict, feature_dim_dict = dict(), dict()
    print("Loading data files..")
    for file_idx, data_file in tqdm(enumerate(data_files)):
        with open(data_file, 'r') as f:
            data = json.load(f)
            for ins_idx, ins in enumerate(data):
                for key in ins.keys():
                    ins[key] = np.array(ins[key]).reshape(len(ins[key]),-1)
                    if file_idx == 0 and ins_idx == 0:
                        print(f'Each {key} shape: {ins[key].shape}')
                        data_len_dict[key] = ins[key].shape[0]
                        feature_dim_dict[key] = ins[key].shape[1]
        if file_idx != len(data_files)-1:
            train_data.extend(data)
        else:
            test_data.extend(data)
    del data
    data_len = data_len_dict['ob']
    pred_len = data_len_dict['tgt'] - data_len_dict['ob']
    
    print("Data files loading completed!")
    print(f'# of train instances: {len(train_data)}')
    print(f'# of test instances: {len(test_data)}')
        
    
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    test_bs = min(len(test_data),args.bs)
    test_loader = DataLoader(test_data, batch_size=test_bs, shuffle=True)
    
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = OrderbookTrade2Price(model_dim=args.model_dim,
                                 n_head=args.n_head,
                                 num_layers=args.num_layers,
                                 ob_feature_dim=feature_dim_dict['ob'],
                                 tr_feature_dim=feature_dim_dict['tr'],
                                 volume_feature_dim=feature_dim_dict['volume'],
                                 tgt_feature_dim=feature_dim_dict['tgt'],
                                 data_len=data_len,
                                 pred_len=pred_len,
                                 ob_importance=args.ob_importance,
                                 tr_importance=args.tr_importance).to(device)

    num_param = 0
    for _, param in model.named_parameters():
        num_param += param.numel()
    print(f'model param size: {num_param}')
    
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in tqdm(range(args.epoch)):
        model.train()
        epoch_loss = 0
        for idx, batch in tqdm(enumerate(train_loader)):
            ob = batch['ob'].to(torch.float32).to(device)
            tr = batch['tr'].to(torch.float32).to(device)
            volume = batch['volume'].to(torch.float32).to(device)
            tgt = torch.clamp(batch['tgt']*args.tgt_amplifier,
                              min=-args.tgt_clip_value, 
                              max=args.tgt_clip_value).to(torch.float32).to(device)
            
            for step in range(pred_len):
                out = model(ob, tr, volume, tgt[:,:data_len+step,:])
                label = tgt[:,1:data_len+step+1,:].squeeze(dim=2)
                loss = loss_function(out,label)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.detach().cpu().item()        
        print(f'Epoch {epoch} Average Loss: {epoch_loss/(idx+1)}')
        scheduler.step()    
    
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_dir)
            # model.load_state_dict(torch.load(save_dir))
            model.eval()
            test_loss = 0
            correct = 0
            rec_correct, rec_tgt = 0,0
            strong_prec_correct, strong_prec_tgt = 0,0
            for idx, batch in tqdm(enumerate(test_loader)):
                ob = batch['ob'].to(torch.float32).to(device)
                tr = batch['tr'].to(torch.float32).to(device)
                volume = batch['volume'].to(torch.float32).to(device)
                tgt = torch.clamp(batch['tgt']*args.tgt_amplifier,
                                  min=-args.tgt_clip_value,
                                  max=args.tgt_clip_value).to(torch.float32).to(device)
                
                for step in range(pred_len):
                    if step == 0:
                        out = model(ob, tr, volume, tgt[:,:data_len,:])
                    else:
                        out = model(ob, tr, volume, torch.cat((tgt[:,:data_len,:], out[:,-step:].unsqueeze(dim=2)),dim=1))

                label = tgt[:,1:,:].squeeze(dim=2)                    
                loss = loss_function(out,label)
                test_loss += loss.detach().cpu().item()
                correct += ((out[:,-1]*label[:,-1])>0).sum().item()

                rec_tgt += (label[:,-1]>=args.value_threshold).to(torch.long).sum().item()
                rec_correct += ((label[:,-1]>=args.value_threshold).to(torch.long) * (out[:,-1]>0).to(torch.long)).sum().item()

                strong_prec_tgt += (out[:,-1]>=args.strong_threshold).to(torch.long).sum().item()
                strong_prec_correct += ((out[:,-1]>=args.strong_threshold).to(torch.long) * (label[:,-1]>0).to(torch.long)).sum().item()
                if idx == 0:
                    print(f'Out: {out[:,-1]}\n Label: {label[:,-1]}')
            print(f'Test Average Loss: {test_loss / (idx+1)}')
            print(f'Test Correct: {correct} out of {test_bs*(idx+1)}')
            print(f'Test Recall: {rec_correct} out of {rec_tgt}')
            print(f'Test Precision (Strong): {strong_prec_correct} out of {strong_prec_tgt}')
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/train/minute')
    parser.add_argument('--save_dir', type=str, default='ckpt/vanilla_minute')
    parser.add_argument('--pred_len', type=int, default=5)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--tgt_amplifier', type=float, default=10)
    parser.add_argument('--tgt_clip_value', type=float, default=1)
    parser.add_argument('--value_threshold', type=float, default=0.5)
    parser.add_argument('--strong_threshold', type=float, default=0.9)
    parser.add_argument('--ob_importance', type=float, default=0.4)
    parser.add_argument('--tr_importance', type=float, default=0.4)
    args = parser.parse_args()
    main(args)