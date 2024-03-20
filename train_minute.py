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

from model.minute import OrderbookTrade2Predictor, OrderbookTrade2Classifier
from utils.train_utils import train_predictor, train_classifier
from utils.label_utils import convert_label


def prepare_data(data_dir, result_dim, value_threshold, tgt_amplifier):
    data_files = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir)])
    train_data, test_data = [], []
    data_len_dict, feature_dim_dict = dict(), dict()
    label_num_list = [0 for _ in range(result_dim)]
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
                label_num_list[convert_label(torch.tensor(ins['tgt'][-1,0]), result_dim, value_threshold/tgt_amplifier).item()] += 1
        if file_idx != len(data_files)-1:
            train_data.extend(data)
        else:
            test_data.extend(data)
    
    return train_data, test_data, feature_dim_dict, data_len_dict, label_num_list


def main(args):
    data_dir = f'{args.data_dir}_{args.pred_len}'
    save_dir = f'{args.save_dir}_{args.train_type}_{args.pred_len}.pt'
    if not os.path.exists(save_dir.split('/')[0]):
        os.makedirs(save_dir.split('/')[0])
    
    train_data, test_data, feature_dim_dict, data_len_dict, label_num_list = prepare_data(data_dir,
                                                                                          args.result_dim,
                                                                                          args.value_threshold,
                                                                                          args.tgt_amplifier)
    
    print("Data files loading completed!")
    print(f'# of train instances: {len(train_data)}')
    print(f'# of test instances: {len(test_data)}')
    print(f'Label Num List: {label_num_list}')

    if args.train_type == 'classifier':
        ce_weight_numerator = len(train_data)+len(test_data)
        ce_weights = torch.tensor([ce_weight_numerator/label_num for label_num in label_num_list])
        ce_weights = nn.functional.normalize(ce_weights, dim=0, p=1)
        print(f'ce_weights: {ce_weights}')

        
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    test_bs = min(len(test_data),args.bs)
    test_loader = DataLoader(test_data, batch_size=test_bs, shuffle=True)
    
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    
    if args.train_type == 'predictor':   
        model = OrderbookTrade2Predictor(model_dim=args.model_dim,
                                         n_head=args.n_head,
                                         num_layers=args.num_layers,
                                         ob_feature_dim=feature_dim_dict['ob'],
                                         tr_feature_dim=feature_dim_dict['tr'],
                                         volume_feature_dim=feature_dim_dict['volume'],
                                         tgt_feature_dim=feature_dim_dict['tgt'],
                                         data_len=data_len_dict['ob'],
                                         pred_len=data_len_dict['tgt'] - data_len_dict['ob'],
                                         ob_importance=args.ob_importance,
                                         tr_importance=args.tr_importance).to(device)

    elif args.train_type == 'classifier':
        model = OrderbookTrade2Classifier(result_dim=args.result_dim,
                                          model_dim=args.model_dim,
                                          n_head=args.n_head,
                                          num_layers=args.num_layers,
                                          ob_feature_dim=feature_dim_dict['ob'],
                                          tr_feature_dim=feature_dim_dict['tr'],
                                          volume_feature_dim=feature_dim_dict['volume'],
                                          tgt_feature_dim=feature_dim_dict['tgt'],
                                          data_len=data_len_dict['ob'],
                                          pred_len=data_len_dict['tgt'] - data_len_dict['ob'],
                                          ob_importance=args.ob_importance,
                                          tr_importance=args.tr_importance).to(device)

    num_param = 0
    for _, param in model.named_parameters():
        num_param += param.numel()
    print(f'model param size: {num_param}')
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.lr*10)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    if args.train_type == 'predictor':
        loss_function = nn.MSELoss()
        train_predictor(model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_function=loss_function,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_bs=test_bs,
                        data_len=data_len_dict['ob'],
                        pred_len=data_len_dict['tgt'] - data_len_dict['ob'],
                        tgt_amplifier=args.tgt_amplifier,
                        tgt_clip_value=args.tgt_clip_value,
                        value_threshold=args.value_threshold,
                        strong_threshold=args.strong_threshold,
                        epoch=args.epoch,
                        device=device,
                        save_dir=save_dir)
        
    elif args.train_type == 'classifier':
        loss_function = nn.CrossEntropyLoss(weight=ce_weights.to(device))
        train_classifier(result_dim=args.result_dim,
                         model=model,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         loss_function=loss_function,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         test_bs=test_bs,
                         data_len=data_len_dict['ob'],
                         pred_len=data_len_dict['tgt'] - data_len_dict['ob'],
                         tgt_amplifier=args.tgt_amplifier,
                         tgt_clip_value=args.tgt_clip_value,
                         value_threshold=args.value_threshold,
                         strong_threshold=args.strong_threshold,
                         epoch=args.epoch,
                         device=device,
                         save_dir=save_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/train/minute')
    parser.add_argument('--save_dir', type=str, default='ckpt/vanilla_minute')
    parser.add_argument('--train_type', type=str, default='predictor', choices=['predictor', 'classifier'])
    parser.add_argument('--pred_len', type=int, default=5)
    parser.add_argument('--result_dim', type=int, default=3)
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
    parser.add_argument('--strong_threshold', type=float, default=0.5)
    parser.add_argument('--ob_importance', type=float, default=0.4)
    parser.add_argument('--tr_importance', type=float, default=0.4)
    args = parser.parse_args()
    main(args)