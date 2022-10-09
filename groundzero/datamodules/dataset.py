"""Parent class for a vision classification dataset."""

# Imports Python builtins
from abc import abstractmethod

# Imports Python packages.
import numpy as np
from PIL import Image

# Imports PyTorch packages.
from torchvision.datasets.vision import VisionDataset


class Dataset(VisionDataset):
    """Parent class for a vision classification dataset.

    Mostly the same as torchvision.VisionDataset with some extra pieces from
    torchvision.CIFAR10Dataset (e.g., the train flag).

    Attributes:
        data: np.ndarray or list containing np.ndarray images or string filenames.
        targets: np.ndarray or list containing classification targets.
        train: Whether the dataset should be loaded in train mode (for transforms, etc.).
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Initializes a Dataset and downloads to disk if requested.

        Args:
            root: The location of the dataset on disk.
            train: Whether the dataset should be loaded in train mode (for transforms, etc.).
            transform: Composition of torchvision.transforms for the image data.
            target_transform: Composition of torchvision.transforms for the targets.
            download: Whether to download the dataset to disk.
        """

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.data = None
        self.targets = None
        self.train = train
        
        if download:
            self.download()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # Loads a PIL.Image either from the cached np.ndarray data or from disk.
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, str):
            img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @abstractmethod
    def download(self):
        pass

