from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

from groundzero.datasets.dataset import Dataset


class CIFAR10(Dataset):
    dataset_class = CIFAR10
    num_classes = 10

    def __init__(self, args):
        super().__init__(args)

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

