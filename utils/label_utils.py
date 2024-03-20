import torch

def convert_label(label, result_dim, value_threshold):
    if result_dim == 2:
        return (label>0).to(torch.long)
    elif result_dim == 3:
        return 1+(label>=value_threshold).to(torch.long)-(label<=-value_threshold).to(torch.long)
    elif result_dim == 4:
        return 1+(label>0).to(torch.long)+(label>=value_threshold).to(torch.long)-(label<=-value_threshold).to(torch.long)
    else:
        raise NotImplementedError(f"Current 2,3,and 4 are only possible values for result_dim, but you have passed {result_dim}")


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