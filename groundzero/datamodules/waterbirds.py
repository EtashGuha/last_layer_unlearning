import os.path as osp
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import  Subset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor

from groundzero.datamodules.dataset import Dataset
from groundzero.datamodules.datamodule import DataModule
from groundzero.utils import to_np


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
        self.landbirds_on_land = np.intersect1d(landbirds, land)
        self.landbirds_on_water = np.intersect1d(landbirds, water)
        self.waterbirds_on_water = np.intersect1d(waterbirds, water)
        self.waterbirds_on_land = np.intersect1d(waterbirds, land)

        split = np.asarray(metadata_df["split"].values)
        self.train_indices = np.argwhere(split == 0).flatten()
        self.val_indices = np.argwhere(split == 1).flatten()
        self.test_indices = np.argwhere(split == 2).flatten()

        # test_group decides which combo of birds/background to test on. 0 is all.
        if test_group == 1:
            self.val_indices = np.intersect1d(self.val_indices, self.landbirds_on_land)
            self.test_indices = np.intersect1d(self.test_indices, self.landbirds_on_land)
        elif test_group == 2:
            self.val_indices = np.intersect1d(self.val_indices, self.landbirds_on_water)
            self.test_indices = np.intersect1d(self.test_indices, self.landbirds_on_water)
        elif test_group == 3:
            self.val_indices = np.intersect1d(self.val_indices, self.waterbirds_on_water)
            self.test_indices = np.intersect1d(self.test_indices, self.waterbirds_on_water)
        elif test_group == 4:
            self.val_indices = np.intersect1d(self.val_indices, self.waterbirds_on_land)
            self.test_indices = np.intersect1d(self.test_indices, self.waterbirds_on_land)

        if not train:
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

    def val_dataloader(self):
        dataloaders = []
        for group in range(5):
            dataloaders.append(
                self._data_loader(
                    self._split_dataset(
                        self.dataset_class(
                            self.data_dir,
                            train=True,
                            transform=self.default_transforms(),
                            test_group=group,
                        ),
                        train=False
            )))

        return dataloaders

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
    def __init__(self, args, model=None, gamma=1, misclassification_dfr=False, full_set_dfr=False, dropout=False, rebalancing=False):
        super().__init__(args)
        self.model = model
        if self.model:
            self.model = self.model.cuda()
        self.disagreement_set = args.disagreement_set
        self.disagreement_proportion = args.disagreement_proportion
        self.gamma = gamma
        self.misclassification_dfr = misclassification_dfr
        self.full_set_dfr = full_set_dfr
        self.dropout = dropout
        self.rebalancing = rebalancing

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

            if self.disagreement_set == "train":
                dataset_reg.train_indices = inds[disagreement_num:]
                dataset_disagreement.train_indices = inds[:disagreement_num]
            elif self.disagreement_set == "val":
                dataset_reg.val_indices = inds[disagreement_num:]
                dataset_disagreement.val_indices = inds[:disagreement_num]

            return dataset_reg, dataset_disagreement
        else:
            return Subset(dataset, inds)

    def disagreement_dataloader(self):
        return self._data_loader(self.dataset_disagreement)

    def rebalance_groups(self, indices, dataset):
        g1 = np.intersect1d(indices, dataset.landbirds_on_land)
        g2 = np.intersect1d(indices, dataset.landbirds_on_water)
        g3 = np.intersect1d(indices, dataset.waterbirds_on_water)
        g4 = np.intersect1d(indices, dataset.waterbirds_on_land)

        m = min(len(g1), len(g2), len(g3), len(g4))

        random.shuffle(g1)
        random.shuffle(g2)
        random.shuffle(g3)
        random.shuffle(g4)

        indices = np.concatenate((g1[:m], g2[:m], g3[:m], g4[:m]))

        return indices

    def rebalance_classes(self, disagree, agree, disagree_targets, agree_targets):
        for i, t in enumerate((disagree_targets, agree_targets)):
            if len(t) == 0:
                continue

            num_one = np.count_nonzero(t)
            num_zero = len(t) - num_one
            to_remove = abs(num_zero - num_one)
            if to_remove == 0:
                continue

            if num_zero > num_one:
                mask = t == 1
            else:
                mask = t == 0

            false_inds = (~mask).nonzero()[0]
            keep = np.random.choice(false_inds, len(false_inds) - to_remove, replace=False)
            mask[keep] = True
            
            if i == 0:
                disagree = disagree[mask]
            elif i == 1:
                agree = agree[mask]

        return disagree, agree

    def disagreement(self):
        dataloader = self.disagreement_dataloader()
        batch_size = dataloader.batch_size

        train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        new_set = self.dataset_class(self.data_dir, train=True, transform=train_transforms)

        disagree = []
        disagree_targets = []
        agree = []
        agree_targets = []

        if self.disagreement_set == "train":
            all_inds = dataloader.dataset.train_indices
        elif self.disagreement_set == "val":
            all_inds = dataloader.dataset.val_indices

        if self.full_set_dfr:
            targets = []
            for batch in dataloader:
                targets.extend(to_np(batch[1]))
            targets = np.asarray(targets)
            indices = all_inds

            if self.rebalancing:
                indices = self.rebalance_groups(indices, new_set)
        else:
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    inputs, targets = batch
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                    if not self.dropout:
                        self.model.eval()
                        logits = self.model(inputs)
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                    else:
                        self.model.eval()
                        orig_logits = self.model(inputs)
                        orig_probs = F.softmax(orig_logits, dim=1)
                        orig_preds = torch.argmax(orig_probs, dim=1)

                        self.model.train()
                        logits = self.model(inputs)
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)

                    if self.misclassification_dfr:
                        disagreements = to_np(torch.logical_xor(preds, targets))
                    elif self.dropout:
                        disagreements = to_np(torch.logical_xor(preds, orig_preds))
                    else:
                        raise ValueError("Can't do disagreement w/o dropout")

                    inds = all_inds[(i * batch_size):min(((i+1) * batch_size), len(all_inds))]
                    disagree.extend(inds[disagreements].tolist())
                    disagree_targets.extend(targets[disagreements].tolist())
                    agree.extend(inds[~disagreements].tolist())
                    agree_targets.extend(targets[~disagreements].tolist())
            
            # Gets a gamma proportion of agreement points.
            if self.gamma > 0:
                num_agree = int(self.gamma * len(disagree))
                c = list(zip(agree, agree_targets))
                random.shuffle(c)
                agree, agree_targets = zip(*c)
                agree = agree[:num_agree]
                agree_targets = agree_targets[:num_agree]
            elif self.gamma < 0: # hack for ablating the disagreement points
                num_agree = int(abs(self.gamma) * len(disagree))
                c = list(zip(agree, agree_targets))
                random.shuffle(c)
                agree, agree_targets = zip(*c)
                agree = agree[:num_agree]
                agree_targets = agree_targets[:num_agree]

                disagree = []
                disagree_targets = []
            else: # gamma == 0
                agree = [] 
                agree_targets = []

            disagree = np.asarray(disagree, dtype=np.int64)
            disagree_targets = np.asarray(disagree_targets, dtype=np.int64)
            agree = np.asarray(agree, dtype=np.int64)
            agree_targets = np.asarray(agree_targets, dtype=np.int64)

            # rebalancing
            # use class labels here
            if self.rebalancing:
                # print the numbers of disagreements by category
                print("Pre-balancing numbers")
                for n, x in zip(("All", "Disagreements", "Agreements"), (all_inds, disagree, agree)):
                    g1 = len(np.intersect1d(x, new_set.landbirds_on_land))
                    g2 = len(np.intersect1d(x, new_set.landbirds_on_water))
                    g3 = len(np.intersect1d(x, new_set.waterbirds_on_water))
                    g4 = len(np.intersect1d(x, new_set.waterbirds_on_land))
                    print(f"{n}: ({g1}, {g2}, {g3}, {g4})")

                disagree, agree = self.rebalance_classes(disagree, agree, disagree_targets, agree_targets)
                            
            indices = np.concatenate((disagree, agree))

        self.dataset_train = Subset(new_set, indices)

        # print the numbers of disagreements by group
        print("Disagreements by group")
        for n, x in zip(("All", "Disagreements", "Agreements", "Total"), (all_inds, disagree, agree, indices)):
            g1 = len(np.intersect1d(x, new_set.landbirds_on_land))
            g2 = len(np.intersect1d(x, new_set.landbirds_on_water))
            g3 = len(np.intersect1d(x, new_set.waterbirds_on_water))
            g4 = len(np.intersect1d(x, new_set.waterbirds_on_land))
            print(f"{n}: ({g1}, {g2}, {g3}, {g4})")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_class(self.data_dir, train=True, transform=train_transforms)
            dataset_val = self.dataset_class(self.data_dir, train=True, transform=val_transforms)

            dataset_train, dataset_val = self.train_preprocess(dataset_train, dataset_val)
            self.dataset_train = self._split_dataset(dataset_train)
            if self.disagreement_set == "train":
                _, self.dataset_disagreement = self._split_dataset(dataset_val, disagreement_proportion=self.disagreement_proportion)
                self.dataset_val = self._split_dataset(dataset_val, train=False)
            elif self.disagreement_set == "val":
                self.dataset_val, self.dataset_disagreement = self._split_dataset(dataset_val, disagreement_proportion=self.disagreement_proportion, train=False)

            # Performs disagreement and sets new train dataset.
            if self.model:
                print("Computing disagreements...")
                self.disagreement()
                del self.model

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            dataset_test = self.dataset_class(self.data_dir, train=False, transform=test_transforms)
            self.dataset_test = self.test_preprocess(dataset_test)
