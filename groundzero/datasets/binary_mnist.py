import torch

from pl_bolts.datasets import MNIST as PLMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from groundzero.datasets.dataset import Dataset


class BinaryMNIST(Dataset):
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
        dataset_train, dataset_val = super().train_preprocess(dataset_train, dataset_val)

        dataset_train.targets = torch.tensor([target % 2 for target in dataset_train.targets])
        dataset_val.targets = torch.tensor([target % 2 for target in dataset_val.targets])

        return dataset_train, dataset_val

    def test_preprocess(self, dataset_test):
        dataset_test = super().test_preprocess(dataset_test)

        dataset_test.targets = torch.tensor([target % 2 for target in dataset_test.targets])

        return dataset_test

