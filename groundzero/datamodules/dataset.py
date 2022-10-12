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

    Mostly the same as torchvision.datasets.VisionDataset with some extra
    pieces from torchvision.datasets.CIFAR10Dataset (e.g., the train flag)
    and support for multiple groups.

    Attributes:
        root: The location of the dataset on disk.
        transform: Composition of torchvision.transforms for the image data.
        target_transform: Composition of torchvision.transforms for the targets.
        data: np.ndarray containing np.ndarray images or string filenames.
        targets: np.ndarray containing classification targets.
        train: Whether the dataset should be loaded in train mode (for transforms, etc.).
        train_indices: Optional np.ndarray of indices of train set.
        val_indices: Optional np.ndarray of indices of val set.
        test_indices: Optional np.ndarray of indices of test set.
        group: If the dataset has multiple groups, specifies which one to initialize.
        groups: If the dataset has multiple groups, lists indices belonging to each group.
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None, 
        download=False,
        group=0,
    ):
        """Initializes a Dataset and downloads to disk if requested.

        If train_indices, etc. are None, then DataModule will do a random split
        based on args.val_split. The np.ndarrays self.data and self.targets
        should hold both training and validation data when train=True and only
        test data when train=False. The split is calculated in the DataModule.

        Args:
            root: The location of the dataset on disk.
            train: Whether the dataset should be loaded in train mode (for transforms, etc.).
            transform: Composition of torchvision.transforms for the image data.
            target_transform: Composition of torchvision.transforms for the targets.
            download: Whether to download the dataset to disk.
            group: If the dataset has multiple groups, specifies which one to initialize.
        """

        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )

        self.train = train
        self.group = group

        self.data = None
        self.targets = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.groups = None

        if download:
            self.download()

        self.load_data()

        if self.groups is not None and self.val_indices is not None:
            self.val_indices = np.intersect1d(self.groups[group], self.val_indices)
        if self.groups is not None and self.test_indices is not None:
            self.test_indices = np.intersect1d(self.groups[group], self.test_indices)

        if self.data is not None and self.targets is not None:
            if self.train_indices is not None and self.val_indices is not None and self.train:
                indices = np.concatenate([self.train_indices, self.val_indices])
                self.data = self.data[indices]
                self.targets = self.targets[indices]
            elif self.test_indices is not None and not self.train:
                self.data = self.data[self.test_indices]
                self.targets = self.targets[self.test_indices]
                
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
        """Downloads dataset to disk."""

    @abstractmethod
    def load_data(self):
        """Initializes self.data and self.targets, and optionally indices and groups."""

