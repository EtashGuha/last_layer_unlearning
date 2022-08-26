from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision.datasets import CIFAR10 as TorchvisionCIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

from groundzero.datamodules.datamodule import DataModule


class CIFAR10(DataModule):
    def __init__(self, args):
        super().__init__(args, TorchvisionCIFAR10, 10)

    def augmented_transforms(self):
        transforms = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            cifar10_normalization(),
        ])

        return transforms

    def default_transforms(self):
        transforms = Compose([
            ToTensor(),
            cifar10_normalization()
        ])

        return transforms
