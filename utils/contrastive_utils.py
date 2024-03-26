import torch
import torch.nn as nn
from .label_utils import get_nondiag_cartesian


def compute_inner_product(out):
    out = nn.functional.normalize(out, dim=1)
    return torch.exp(torch.matmul(out, out.transpose(0,1))/out.shape[1])


def compute_contrastive_logits(out, idx):
    out_sum = out.sum(axis=1) - torch.diagonal(out)
    idx_pairs = get_nondiag_cartesian(idx)
    
    if len(idx)>1:
        logit_sum = out_sum[idx]
        positive_logit_sum = out[idx_pairs[0], idx_pairs[1]].reshape(-1,len(idx)-1).sum(axis=1)
        return -torch.mean(torch.log(positive_logit_sum) - torch.log(logit_sum))
    
    else:
        return torch.tensor([0]).to(out.device)


def compute_contrastive_loss(out, leftmost_label_idx, rightmost_label_idx, contrastive_side):
    if contrastive_side != 'right':
        leftmost_logit = compute_contrastive_logits(out, leftmost_label_idx)
    if contrastive_side != 'left':
        rightmost_logit = compute_contrastive_logits(out, rightmost_label_idx)

    if contrastive_side == 'left' and len(leftmost_label_idx)>1:
        loss = leftmost_logit
    elif contrastive_side == 'right' and len(rightmost_label_idx)>1:
        loss = rightmost_logit
    elif contrastive_side == 'both' and max(len(leftmost_label_idx),len(rightmost_label_idx))>1:
        loss = leftmost_logit + rightmost_logit
    else:
        loss = None
    
    return loss