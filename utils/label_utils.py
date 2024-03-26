import torch
import torch.nn.functional as F

def convert_label(label, result_dim, value_threshold):
    if result_dim == 2:
        return (label>0).to(torch.long)
    elif result_dim == 3:
        return 1+(label>=value_threshold).to(torch.long)-(label<=-value_threshold).to(torch.long)
    elif result_dim == 4:
        return 1+(label>0).to(torch.long)+(label>=value_threshold).to(torch.long)-(label<=-value_threshold).to(torch.long)
    else:
        raise NotImplementedError(f"Current 2,3,and 4 are only possible values for result_dim, but you have passed {result_dim}")


def get_one_hot_label(label, result_dim, value_threshold):
    label = convert_label(label, result_dim, value_threshold)
    return F.one_hot(label, num_classes=result_dim).to(torch.float32)


def is_strong_label(label, result_dim):
    if result_dim == 3:
        return (label!=1).to(torch.long)
    elif result_dim == 4:
        return 1-((label>=1).to(torch.long) * (label<=2).to(torch.long))
    else:
        raise NotImplementedError(f"Current version of checking if strong label is supported only for result_dim of 3 and 4, but you have passed {result_dim}")


def is_close_pred(pred, label, result_dim):
    if result_dim == 3:
        close_threshold = 1
    elif result_dim == 4:
        close_threshold = 1  
    else:
        raise NotImplementedError(f"Current version of checking if pred and label are close is supported only for result_dim of 3 and 4, but you have passed {result_dim}")
    return (torch.abs(pred-label) <= close_threshold).to(torch.long)


def get_extreme_label_pairs(label, result_dim):
    leftmost_label_idx = (label==0).nonzero().squeeze(dim=1)
    rightmost_label_idx = (label==result_dim-1).nonzero().squeeze(dim=1)
    
    return leftmost_label_idx, rightmost_label_idx


def get_nondiag_cartesian(idx):
    idx_pairs = torch.cartesian_prod(idx, idx).transpose(0,1)
    nondiag_mask = (1-torch.eye(len(idx))).reshape(-1).nonzero().squeeze(dim=1)
    return idx_pairs[:,nondiag_mask]