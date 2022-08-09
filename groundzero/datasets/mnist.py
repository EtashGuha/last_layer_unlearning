from pl_bolts.datasets import MNIST as PLMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from groundzero.datasets.dataset import Dataset


class MNIST(Dataset):
    def __init__(self, args):
        super().__init__(args, PLMNIST, 10)

    def augmented_transforms(self):
        return self.default_transforms()

    def default_transforms(self):
        transforms = Compose([
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,))
        ])

        return transforms

