import torch

import numpy as np

def compute_accuracy(probs, targets):
    preds1 = torch.argmax(probs, dim=1)
    _, preds5 = torch.topk(probs, k=5, dim=1)

    acc1 = (preds1 == targets).float().mean()
    acc5 = torch.tensor(
        [t in preds5[j] for j, t in enumerate(targets)],
        dtype=torch.float64,
    ).mean()

    return acc1, acc5

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.asarray(x)
    else:
        raise ValueError("Undefined.")
