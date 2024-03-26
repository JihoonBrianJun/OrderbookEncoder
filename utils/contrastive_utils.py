import torch
from .label_utils import get_nondiag_cartesian


def compute_contrastive_logits(out, idx):
    out_sum = out.sum(axis=1) - torch.diagonal(out)
    idx_pairs = get_nondiag_cartesian(idx)
    
    if len(idx)>1:
        logit_sum = out_sum[idx]
        positive_logit_sum = out[idx_pairs[0], idx_pairs[1]].reshape(-1,len(idx)-1).sum(axis=1)
        return -torch.mean(torch.log(positive_logit_sum) - torch.log(logit_sum))
    
    else:
        return torch.tensor([0]).to(out.device)