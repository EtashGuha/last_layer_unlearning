import os.path as osp

import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import  Subset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor

from groundzero.datamodules.dataset import Dataset
from groundzero.datamodules.datamodule import DataModule


class WaterbirdsDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        waterbirds_dir = osp.join(root, "waterbirds")
        metadata_df = pd.read_csv(osp.join(waterbirds_dir, "metadata.csv"))
        imgs = np.asarray(metadata_df["img_filename"].values)
        self.targets = np.asarray(metadata_df["y"].values)

        self.data = []
        for img in imgs:
            img_path = osp.join(waterbirds_dir, img)
            self.data.append(Image.open(img_path))

        split = np.asarray(metadata_df["split"].values)
        self.train_indices = np.argwhere(split == 0)
        self.val_indices = np.argwhere(split == 1)
        self.test_indices = np.argwhere(split == 2)

        if not train:
            self.data = [d for j, d in enumerate(self.data) if j in self.test_indices]
            self.targets = self.targets[self.test_indices]

    def download(self):
        waterbirds_dir = osp.join(self.root, "waterbirds")
        if not osp.isdir(waterbirds_dir):
            download_and_extract_archive(
                "http://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/",
                waterbirds_dir,
                filename="waterbirds.tar.gz",
            )

class Waterbirds(DataModule):
    def __init__(self, args):
        super().__init__(args, WaterbirdsDataset, 2)

    def load_msg(self):
        msg = f"Loading {type(self).__name__} with default val split."

        if self.data_augmentation:
            msg = msg[:-1] + " and data augmentation."
        if self.label_noise:
            msg = msg[:-1] + f" and {int(self.label_noise * 100)}% label noise."

        return msg

    def _split_dataset(self, dataset, train=True):
        dataset_train = Subset(dataset, dataset.train_indices)
        dataset_val = Subset(dataset, dataset.val_indices)

        if train:
            return dataset_train
        return dataset_val

    def augmented_transforms(self):
        transform = Compose([
            RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return transform

    def default_transforms(self):
        transform = Compose([
            Resize((256, 256)),
            CenterCrop((224, 244)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return transform

