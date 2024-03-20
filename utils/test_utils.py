import torch
from tqdm import tqdm
from .label_utils import convert_label, is_strong_label, is_close_pred

def test_predictor(model, loss_function,
                   dataloader, test_bs,
                   data_len, pred_len, tgt_amplifier, tgt_clip_value,
                   value_threshold, strong_threshold,
                   device, save_dir, save_ckpt=True, load_ckpt=False):
    
    if save_ckpt:
        torch.save(model.state_dict(), save_dir)
    if load_ckpt:
        model.load_state_dict(torch.load(save_dir))

    model.eval()
    test_loss = 0
    correct = 0
    rec_correct, rec_tgt = 0,0
    strong_prec_correct, strong_prec_tgt = 0,0
    for idx, batch in tqdm(enumerate(dataloader)):
        ob = batch['ob'].to(torch.float32).to(device)
        tr = batch['tr'].to(torch.float32).to(device)
        volume = batch['volume'].to(torch.float32).to(device)
        tgt = torch.clamp(batch['tgt']*tgt_amplifier,
                          min=-tgt_clip_value,
                          max=tgt_clip_value).to(torch.float32).to(device)
        
        for step in range(pred_len):
            if step == 0:
                out = model(ob, tr, volume, tgt[:,:data_len,:])
            else:
                out = model(ob, tr, volume, torch.cat((tgt[:,:data_len,:], out[:,-step:].unsqueeze(dim=2)),dim=1))

        label = tgt[:,1:,:].squeeze(dim=2)                    
        loss = loss_function(out,label)
        test_loss += loss.detach().cpu().item()
        correct += ((out[:,-1]*label[:,-1])>0).sum().item()

        rec_tgt += (label[:,-1]>=value_threshold).to(torch.long).sum().item()
        rec_correct += ((label[:,-1]>=value_threshold).to(torch.long) * (out[:,-1]>0).to(torch.long)).sum().item()

        strong_prec_tgt += (out[:,-1]>=strong_threshold).to(torch.long).sum().item()
        strong_prec_correct += ((out[:,-1]>=strong_threshold).to(torch.long) * (label[:,-1]>0).to(torch.long)).sum().item()
        
        if idx == 0:
            print(f'Out: {out[:,-1]}\n Label: {label[:,-1]}')
            
    print(f'Test Average Loss: {test_loss / (idx+1)}')
    print(f'Test Correct: {correct} out of {test_bs*(idx+1)}')
    print(f'Test Recall: {rec_correct} out of {rec_tgt}')
    print(f'Test Precision (Strong): {strong_prec_correct} out of {strong_prec_tgt}')


def test_classifier(result_dim, model, loss_function,
                    dataloader, test_bs,
                    data_len, pred_len, tgt_amplifier, tgt_clip_value,
                    value_threshold, strong_threshold,
                    device, save_dir, save_ckpt=True, load_ckpt=False):
    
    if save_ckpt:
        torch.save(model.state_dict(), save_dir)
    if load_ckpt:
        model.load_state_dict(torch.load(save_dir))

    model.eval()
    test_loss = 0
    correct = 0
    prec_correct, prec_tgt = 0,0
    rec_correct, rec_tgt = 0,0
    strong_prec_correct, strong_prec_tgt = 0,0
    prec_close, strong_prec_close = 0,0
    for idx, batch in tqdm(enumerate(dataloader)):
        ob = batch['ob'].to(torch.float32).to(device)
        tr = batch['tr'].to(torch.float32).to(device)
        volume = batch['volume'].to(torch.float32).to(device)
        tgt = torch.clamp(batch['tgt']*tgt_amplifier,
                          min=-tgt_clip_value,
                          max=tgt_clip_value).to(torch.float32).to(device)
        
        for step in range(pred_len):
            if step == 0:
                out = model(ob, tr, volume, tgt[:,:data_len,:])
            else:
                out = model(ob, tr, volume, torch.cat((tgt[:,:data_len,:], out[:,-step:].unsqueeze(dim=2)),dim=1))

        label = tgt[:,1:,:].squeeze(dim=2)
        label = convert_label(label, result_dim, value_threshold) 
                    
        loss = loss_function(out.view(-1,result_dim),label.view(-1))
        test_loss += loss.detach().cpu().item()
        correct += (torch.argmax(out[:,-1,:],dim=1)==label[:,-1]).sum().item()

        rec_tgt += is_strong_label(label[:,-1], result_dim).sum().item()
        rec_correct += (is_strong_label(label[:,-1], result_dim) * (torch.argmax(out[:,-1,:],dim=1)==label[:,-1]).to(torch.long)).sum().item()

        prec_tgt += is_strong_label(torch.argmax(out[:,-1,:],dim=1), result_dim).sum().item()
        prec_correct += (is_strong_label(torch.argmax(out[:,-1,:],dim=1), result_dim) * (torch.argmax(out[:,-1,:],dim=1)==label[:,-1]).to(torch.long)).sum().item()

        strong_prec_tgt += ((torch.max(out[:,-1,:],dim=1).values>=strong_threshold).to(torch.long) * is_strong_label(torch.argmax(out[:,-1,:],dim=1), result_dim)).sum().item()
        strong_prec_correct += ((torch.max(out[:,-1,:],dim=1).values>=strong_threshold).to(torch.long) * is_strong_label(torch.argmax(out[:,-1,:],dim=1), result_dim) * (torch.argmax(out[:,-1,:],dim=1)==label[:,-1]).to(torch.long)).sum().item()

        prec_close += (is_strong_label(torch.argmax(out[:,-1,:],dim=1), result_dim) * is_close_pred(torch.argmax(out[:,-1,:],dim=1), label[:,-1], result_dim)).sum().item()
        strong_prec_close += ((torch.max(out[:,-1,:],dim=1).values>=strong_threshold).to(torch.long) * is_strong_label(torch.argmax(out[:,-1,:],dim=1), result_dim) * is_close_pred(torch.argmax(out[:,-1,:],dim=1), label[:,-1], result_dim)).sum().item()

        if idx == 0:
            print(f'Out: {out[:,-1,:]}\n Label: {label[:,-1]}')
            
    print(f'Test Average Loss: {test_loss / (idx+1)}')
    print(f'Test Correct: {correct} out of {test_bs*(idx+1)}')
    print(f'Test Recall: {rec_correct} out of {rec_tgt}')
    print(f'Test Precision: {prec_correct} out of {prec_tgt}')
    print(f'Test Precision (Strong): {strong_prec_correct} out of {strong_prec_tgt}')
    print(f'Test Precision_Close: {prec_close} out of {prec_tgt}')
    print(f'Test Precision_Close (Strong): {strong_prec_close} out of {strong_prec_tgt}')