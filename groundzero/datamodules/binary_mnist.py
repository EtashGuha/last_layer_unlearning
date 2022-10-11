"""DataModule for the BinaryMNIST dataset."""

# Imports PyTorch packages.
from pl_bolts.datasets import MNIST as PLMNIST
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

# Imports groundzero packages.
from groundzero.datamodules.datamodule import DataModule


class BinaryMNIST(DataModule):
    """DataModule for the BinaryMNIST dataset.

    The BinaryMNIST dataset uses the same data as the MNIST dataset, but
    turns it into a binary classification task between odd and even digits.
    """

    def __init__(self, args):
        super().__init__(args, PLMNIST, 2)

    def augmented_transforms(self):
        return self.default_transforms()

    def default_transforms(self):
        transforms = Compose([
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,))
        ])

        return transforms

    def train_preprocess(self, dataset_train, dataset_val):
        dataset_train.targets = torch.tensor([target % 2 for target in dataset_train.targets])
        dataset_val.targets = torch.tensor([target % 2 for target in dataset_val.targets])
        dataset_train, dataset_val = super().train_preprocess(dataset_train, dataset_val)
        return dataset_train, dataset_val

    def test_preprocess(self, dataset_test):
        dataset_test.targets = torch.tensor([target % 2 for target in dataset_test.targets])
        dataset_test = super().test_preprocess(dataset_test)
        return dataset_test

