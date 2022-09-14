import os.path as osp
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import  Subset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor

from groundzero.datamodules.dataset import Dataset
from groundzero.datamodules.datamodule import DataModule


class WaterbirdsDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, test_group=0):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        waterbirds_dir = osp.join(root, "waterbirds")
        metadata_df = pd.read_csv(osp.join(waterbirds_dir, "metadata.csv"))
        self.data = np.asarray(metadata_df["img_filename"].values)
        self.data = np.asarray([osp.join(waterbirds_dir, d) for d in self.data])

        self.targets = np.asarray(metadata_df["y"].values)
        background = np.asarray(metadata_df["place"].values)
        landbirds = np.argwhere(self.targets == 0).flatten()
        waterbirds = np.argwhere(self.targets == 1).flatten()
        land = np.argwhere(background == 0).flatten()
        water = np.argwhere(background == 1).flatten()
        landbirds_on_land = np.intersect1d(landbirds, land)
        waterbirds_on_water = np.intersect1d(waterbirds, water)
        landbirds_on_water = np.intersect1d(landbirds, water)
        waterbirds_on_land = np.intersect1d(waterbirds, land)

        split = np.asarray(metadata_df["split"].values)
        self.train_indices = np.argwhere(split == 0).flatten()
        self.val_indices = np.argwhere(split == 1).flatten()
        self.test_indices = np.argwhere(split == 2).flatten()

        if not train:
            # test_group decides which combo of birds/background to test on. 0 is all.
            if test_group == 1:
                self.test_indices = np.intersect1d(self.test_indices, landbirds_on_land)
            elif test_group == 2:
                self.test_indices = np.intersect1d(self.test_indices, waterbirds_on_water)
            elif test_group == 3:
                self.test_indices = np.intersect1d(self.test_indices, landbirds_on_water)
            elif test_group == 4:
                self.test_indices = np.intersect1d(self.test_indices, waterbirds_on_land)

            self.data = self.data[self.test_indices]
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

    def test_dataloader(self):
        dataloaders = []
        for group in range(5):
            dataloaders.append(
                self._data_loader(
                    self.dataset_class(
                        self.data_dir,
                        train=False,
                        transform=self.default_transforms(),
                        test_group=group,
            )))

        return dataloaders

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
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return transform

class WaterbirdsDisagreement(Waterbirds):
    def __init__(self, args, disagreement_set=None, disagreement_proportion=None, student=None, lmbda=None):
        super().__init__(args)
        self.disagreement_set = disagreement_set
        self.disagreement_proportion = disagreement_proportion
        self.student = student
        self.lmbda = lmbda

        if disagreement_set == "train":
            raise ValueError("did not implement transforms yet, use val")

    def load_msg(self):
        msg = super().load_msg()
        msg = msg[:-1] + f", with disagreement set {self.disagreement_set} and proportion {self.disagreement_proportion}."
        return msg

    def _split_dataset(self, dataset, disagreement_proportion=None, train=True):
        if train:
            inds = dataset.train_indices
        else:
            inds = dataset.val_indices

        if disagreement_proportion:
            random.shuffle(inds)
            disagreement_num = int(disagreement_proportion * len(inds))

            dataset_reg = Subset(dataset, inds[disagreement_num:])
            dataset_disagreement = Subset(dataset, inds[:disagreement_num])

            return dataset_reg, dataset_disagreement
        else:
            return Subset(dataset, inds)

    def disagreement_dataloader(self):
        return self._data_loader(self.dataset_disagreement, shuffle=False)

    def disagreement(self):
        self.student.eval()
        dataloader = self.disagreement_dataloader()
        batch_size = dataloader.batch_size

        indices = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs, _ = batch

                logits = self.student(inputs)
                teacher_logits = self.student.teacher(inputs)

                preds = torch.sigmoid(logits) > 0.5
                teacher_preds = torch.sigmoid(teacher_logits) > 0.5

                # add disagreements
                disagreements = torch.logical_xor(preds, teacher_preds).tolist()
                inds = [j + (i * batch_size) for j, d in enumerate(disagreements) if d == True]
                indices.extend(inds)

                # add specified proportion of agreements
                inds = [j + (i * batch_size) for j, d in enumerate(disagreements) if d == False]
                num_agreements = int(self.lmbda / 1 - self.lmbda)
                if num_agreements > len(inds):
                    raise ValueError("Lambda is too small")
                inds = inds[:num_agreements]
                indices.extend(inds)

        # needs to be changed for disagreement set train
        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        val = self.dataset_class(self.data_dir, train=True, transform=train_transforms)
        _, disagreement = self._split_dataset(val, disagreement_proportion=self.disagreement_proportion, train=False)
        self.dataset_train = Subset(disagreement, indices)

        # print the numbers of disagreements by category

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_class(self.data_dir, train=True, transform=train_transforms)
            dataset_val = self.dataset_class(self.data_dir, train=True, transform=val_transforms)

            dataset_train, dataset_val = self.train_preprocess(dataset_train, dataset_val)
            if self.disagreement_set == "train":
                # here we actually want no transforms on the disagreement set?
                self.dataset_train, self.dataset_disagreement = self._split_dataset(dataset_train, disagreement_proportion=self.disagreement_proportion)
                self.dataset_val = self._split_dataset(dataset_val, train=False)
            elif self.disagreement_set == "val":
                self.dataset_train = self._split_dataset(dataset_train)
                self.dataset_val, self.dataset_disagreement = self._split_dataset(dataset_val, disagreement_proportion=self.disagreement_proportion, train=False)

            # perform disagreement and set new train dataset
            if self.student:
                print("Computing disagreements...")
                self.disagreement()
                del self.student

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            dataset_test = self.dataset_class(self.data_dir, train=False, transform=test_transforms)
            self.dataset_test = self.test_preprocess(dataset_test)
