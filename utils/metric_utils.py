import torch
from .label_utils import is_strong_label, is_close_pred


def compute_predictor_metrics(pred, target, value_threshold, strong_threshold):
    metrics = dict()
    metrics["correct"] = ((pred*target)>0).sum().item()

    metrics["rec_tgt"] = (target>=value_threshold).to(torch.long).sum().item()
    metrics["rec_correct"] = ((target>=value_threshold).to(torch.long) * (pred>0).to(torch.long)).sum().item()

    metrics["strong_prec_tgt"] = (pred>=strong_threshold).to(torch.long).sum().item()
    metrics["strong_prec_correct"] = ((pred>=strong_threshold).to(torch.long) * (target>0).to(torch.long)).sum().item()

    return metrics


def compute_classifier_metrics(pred, target, result_dim, strong_threshold):
    metrics = dict()
    pred_argmax = torch.argmax(pred,dim=1)
    pred_max = torch.max(pred,dim=1)
    
    metrics["correct"] = (pred_argmax==target).sum().item()

    metrics["rec_tgt"] = is_strong_label(target,result_dim).sum().item()
    metrics["rec_correct"] = (is_strong_label(target,result_dim) * (pred_argmax==target).to(torch.long)).sum().item()

    metrics["prec_tgt"] = is_strong_label(pred_argmax,result_dim).sum().item()
    metrics["prec_correct"] = (is_strong_label(pred_argmax,result_dim) * (pred_argmax==target).to(torch.long)).sum().item()

    metrics["strong_prec_tgt"] = ((pred_max.values>=strong_threshold).to(torch.long) * is_strong_label(pred_argmax,result_dim)).sum().item()
    metrics["strong_prec_correct"] = ((pred_max.values>=strong_threshold).to(torch.long) * is_strong_label(pred_argmax,result_dim) * (pred_argmax==target).to(torch.long)).sum().item()

    metrics["prec_close"] = (is_strong_label(pred_argmax,result_dim) * is_close_pred(pred_argmax,target,result_dim)).sum().item()
    metrics["strong_prec_close"] = ((pred_max.values>=strong_threshold).to(torch.long) * is_strong_label(pred_argmax,result_dim) * is_close_pred(pred_argmax,target,result_dim)).sum().item()
    
    return metrics


def compute_contrastive_metrics(out, leftmost_label_idx, rightmost_label_idx):
    metrics = dict()

    if len(leftmost_label_idx) == 1:
        metrics["leftmost_tgt"] = 0
    else:
        metrics["leftmost_tgt"] = len(leftmost_label_idx)
    
    if len(rightmost_label_idx) == 1:
        metrics["rightmost_tgt"] = 0
    else:
        metrics["rightmost_tgt"] = len(rightmost_label_idx)
    
    out_leftmost_argmax = torch.sort(out[leftmost_label_idx], dim=1).indices[:,-2]
    out_rightmost_argmax = torch.sort(out[rightmost_label_idx], dim=1).indices[:,-2]

    metrics["leftmost_correct"] = torch.isin(out_leftmost_argmax, leftmost_label_idx).sum().item()
    metrics["rightmost_correct"] = torch.isin(out_rightmost_argmax, rightmost_label_idx).sum().item()
    
    return metrics