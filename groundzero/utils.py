import torch

import numpy as np


def compute_accuracy(probs, targets, num_classes):
    # TODO: Support targets not indexed by range(num_classes).

    if num_classes == 1:
        preds1 = (probs >= 0.5).int()
        acc1 = (preds1 == targets).float().mean()
    else:
        preds1 = torch.argmax(probs, dim=1)
        acc1 = (preds1 == targets).float().mean()

    acc5 = 1.0
    if num_classes >= 5:
        _, preds5 = torch.topk(probs, k=5, dim=1)
        acc5 = torch.tensor(
            [t in preds5[j] for j, t in enumerate(targets)],
            dtype=torch.float64,
        ).mean()

    return acc1, acc5

def _to_np(x):
    return x.cpu().detach().numpy()

def to_np(x):
    if isinstance(x, torch.Tensor):
        return _to_np(x)
    elif isinstance(x, (np.ndarray, list)):
        if isinstance(x[0], torch.Tensor):
            return _to_np(torch.tensor(x))
        else:
            return np.asarray(x)
    else:
        raise ValueError("Undefined.")

