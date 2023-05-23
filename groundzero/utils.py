"""Utility functions for groundzero."""

# Imports Python packages.
import math
import numpy as np

# Imports PyTorch packages.
import torch
from torch._utils import _accumulate 

# Imports groundzero packages.
from groundzero.datamodules.dataset import Subset


def compute_accuracy(probs, targets, num_classes):
    """Computes top-1 and top-5 accuracies.

    Args:
        probs: A torch.Tensor of prediction probabilities.
        targets: A torch.Tensor of classification targets.
        num_classes: The total number of classes.

    Returns:
        The top-1, top-1 by class, top-5, and top-5 by class accuracies of probs on targets.
    """

    # TODO: Support targets not indexed by range(num_classes).

    if num_classes == 1:
        preds1 = (probs >= 0.5).int()
    else:
        preds1 = torch.argmax(probs, dim=1)

    correct = preds1 == targets
    acc1 = correct.float().mean()

    acc1_by_class = []
    for j in range(num_classes):
        acc1_by_class.append(correct[targets == j].float().mean())
    acc1_by_class = torch.stack(acc1_by_class)

    acc5 = torch.tensor(1.)
    acc5_by_class = torch.tensor([1. for _ in range(num_classes)])
    if num_classes > 5:
        _, preds5 = torch.topk(probs, k=5, dim=1)
        correct = torch.tensor([t in preds5[j] for j, t in enumerate(targets)])
        acc5 = correct.float().mean()

        acc5_by_class = []
        for j in range(num_classes):
            acc5_by_class.append(correct[targets == j].float().mean())
        acc5_by_class = torch.stack(acc5_by_class)

    return acc1, acc1_by_class, acc5, acc5_by_class

def _to_np(x):
    return x.cpu().detach().numpy()

def to_np(x):
    """Converts input to numpy array.

    Args:
        x: A torch.Tensor, np.ndarray, or list.

    Returns:
        The input converted to a numpy array.

    Raises:
        ValueError: The input cannot be converted to a numpy array.
    """

    if not len(x):
        return np.array([])
    elif isinstance(x, torch.Tensor):
        return _to_np(x)
    elif isinstance(x, (np.ndarray, list)):
        if isinstance(x[0], torch.Tensor):
            return _to_np(torch.tensor(x))
        else:
            return np.asarray(x)
    else:
        raise ValueError("Undefined input.")

def random_split(dataset, lengths, generator):
    """Random split function from PyTorch adjusted for groundzero.Subset."""

    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)

        # Adds 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                print(f"Length of split at index {i} is 0. "
                      f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator).tolist() 
    return [Subset(dataset, indices[offset - length : offset])
            for offset, length in zip(_accumulate(lengths), lengths)]

