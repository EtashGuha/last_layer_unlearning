from abc import abstractmethod

import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset


class Dataset(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
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

