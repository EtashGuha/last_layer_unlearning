"""DataModule for disagreement-based datasets."""

# Imports Python builtins.
from math import ceil
import random

# Imports Python packages.
import numpy as np

# Imports PyTorch packages.
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

# Imports groundzero packages.
from groundzero.datamodules.dataset import Subset
from groundzero.datamodules.datamodule import DataModule
from groundzero.utils import to_np


class Disagreement(DataModule):
    """DataModule for disagreement sets used for deep feature reweighting (DFR).

    The original DFR procedure uses group annotations to construct a reweighting
    dataset that has equal data from every group. We propose using disagreement
    between the ERM model and a resource-constrained (e.g., dropout) model
    as an alternative. This enables construction of a nearly-group-balanced
    reweighting dataset without the need for group annotations.

    This class is currently only defined for datasets with a preset val
    split, i.e., datasets which have train_indices and val_indices.

    Attributes:
        model: The groundzero.models.Model used for disagreement.
        disagreement_proportion: The proportion of the dataset to use for disagreement.
        gamma: The proportion of agreements with which to augment disagreements.
        orig_dfr: Whether to use group labels to perform original DFR.
        misclassification_dfr: Whether to perform misclassification DFR.
        dropout_dfr: Whether to perform dropout DFR.
        disagreement_ablation: Whether to only use agreement points.
        dataset_disagreement: A groundzero.datamodules.Dataset for disagreement.
    """

    def __init__(
        self,
        args,
        *xargs,
        num_data=None,
        model=None,
        earlystop_model=None,
        kl_ablation=None,
        gamma=None,
        class_balancing=False,
    ):
        """Initializes a Disagreement DataModule.
        
        Args:
            args:
            *xargs:
            model: The groundzero.models.Model used for disagreement.
        """

        super().__init__(args, *xargs)
        
        self.disagreement_proportion = 0.5
        self.dfr_type = args.dfr_type if hasattr(args, "dfr_type") else None
        self.combine_val_set = args.combine_val_set if hasattr(args, "combine_val_set") else False
        self.num_data = num_data
        self.model = model.cuda() if model else None
        self.earlystop_model = earlystop_model.cuda() if earlystop_model else None
        self.kl_ablation = kl_ablation
        self.class_balancing = class_balancing
        self.gamma = gamma

    def load_msg(self):
        """Returns a descriptive message about the DataModule configuration."""

        msg = super().load_msg()
        msg = msg[:-1] + (
            f", with proportion {self.disagreement_proportion}."
        )
        return msg

    def _split_dataset(self, dataset, disagreement_proportion=None, train=True):
        """Splits dataset into training and validation subsets.

        Args:
            dataset: A groundzero.datamodules.Dataset.
            disagreement_proportion: The proportion of the dataset for disagreement.

        Returns:
        """

        inds = dataset.train_indices if train else dataset.val_indices
        if train and not self.combine_val_set:
            return Subset(dataset, inds)

        if train and disagreement_proportion and self.combine_val_set:
            val_inds = dataset.val_indices
            random.shuffle(val_inds)
            disagreement_num = int(disagreement_proportion * len(val_inds))

            inds = np.concatenate((dataset.train_indices, val_inds[:disagreement_num]))
            dataset.train_indices = inds
            dataset.val_indices = val_inds[disagreement_num:]
            dataset_combined = Subset(dataset, inds)

            return dataset_combined

        if disagreement_proportion:
            if float(disagreement_proportion) != 1.:
                random.shuffle(inds)
                disagreement_num = int(disagreement_proportion * len(inds))

                dataset_reg = Subset(dataset, inds[disagreement_num:])
                dataset_disagreement = Subset(dataset, inds[:disagreement_num])

                dataset_reg.val_indices = inds[disagreement_num:]
                dataset_disagreement.val_indices = inds[:disagreement_num]

                return dataset_reg, dataset_disagreement
            else:
                return Subset(dataset, inds), Subset(dataset, inds)
        else:
            return Subset(dataset, inds)

    def train_dataloader(self):
        """Returns DataLoader for the train dataset (after disagreement)."""

        if self.dfr_type == "orig":
            # Does group balancing for original DFR.
            indices = self.dataset_train.train_indices
            groups = np.zeros(len(indices), dtype=np.int32)
            for i, x in enumerate(indices):
                for j, group in enumerate(self.dataset_train.groups[1:]):
                    if x in group:
                        groups[i] = j

            counts = np.bincount(groups)
            label_weights = 1. / counts
            weights = label_weights[groups]
            sampler = WeightedRandomSampler(weights, len(weights))
            return self._data_loader(self.dataset_train, sampler=sampler)
        elif self.class_balancing:
            self.balanced_sampler = True
            return super().train_dataloader()
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        dataloaders = super().val_dataloader()
        return dataloaders[1:] # remove group 0
        #return dataloaders

    def test_dataloader(self):
        dataloaders = super().test_dataloader()
        return dataloaders[1:] # remove group 0
        #return dataloaders

    def disagreement_dataloader(self):
        """Returns DataLoader for the disagreement set."""

        return self._data_loader(self.dataset_disagreement)

    def print_disagreements_by_group(self, dataset, all_inds, disagree, agree, indices=None):
        """Prints number of disagreements or agreements occuring in each group.
        
        Args:
            dataset: A groundzero.datamodules.Dataset.
            all_inds: An np.ndarray of all indices in the disagreement set.
            disagree: An np.ndarray of all disagreed indices.
            agree: An np.ndarray of all agreed indices.
            indices: An optional np.ndarray of final indices used for DFR.
        """
        if indices is not None:
            labels_and_inds = zip(
                ("All", "Disagreements", "Agreements", "Final DFR Set"),
                (all_inds, disagree, agree, indices),
            )
        else:
            labels_and_inds = zip(
                ("All", "Disagreements", "Agreements"),
                (all_inds, disagree, agree),
            )

        print("Disagreements by group")
        for label, inds in labels_and_inds:
            # Doesn't print for group 0, by convention the group of all indices.
            nums = []
            for group in dataset.groups[1:]:
                nums.append(len(np.intersect1d(inds, group)))
            print(f"{label}: {nums}")
    
    def disagreement(self):
        """Computes disagreement set and saves it as self.dataset_train.
        
        self.dataset_disagreement is initially the self.disagreement_proportion
        of the held-out set. Here, we perform some computation (i.e., the actual
        disagreement) on self.dataset_disagreement to get indices for DFR. Then,
        we set these indices as self.dataset_train for DFR training.
        """

        dataloader = self.disagreement_dataloader()
        batch_size = dataloader.batch_size

        new_set = self.dataset_class(
            self.data_dir,
            train=True,
            transform=self.train_transforms,
        )

        disagree = []
        disagree_targets = []
        agree = []
        agree_targets = []

        all_inds = dataloader.dataset.val_indices

        if self.dfr_type in ("orig", "random"):
            targets = []
            for batch in dataloader:
                targets.extend(to_np(batch[1]))

            if self.num_data == "all":
                p = len(all_inds)
            else:
                p = self.num_data

            inds = np.random.choice(
                np.arange(len(all_inds)),
                #size=int(ceil(len(all_inds)*self.num_data)),
                size=p,
                replace=False,
            )
            indices = all_inds[inds]
            targets = to_np(targets)[inds]

            """
            minority = np.intersect1d(indices, new_set.groups[4])
            del_inds = [j for j, x in enumerate(indices) if x in minority[3:]]
            indices = np.delete(indices, del_inds)
            targets = np.delete(targets, del_inds)
            """
           
        else:
            all_orig_logits = []
            all_logits = []
            all_orig_probs = []
            all_probs = []
            all_targets = []

            # Performs misclassification or dropout disagreements with self.model.
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    inputs, targets = batch
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                    if self.dfr_type in ("earlystop_dropout", "earlystop_miscls"):
                        self.earlystop_model.eval()
                        orig_logits = self.earlystop_model(inputs)
                        orig_probs = F.softmax(orig_logits, dim=1)
                        orig_preds = torch.argmax(orig_probs, dim=1)
                    else:
                        self.model.eval()
                        orig_logits = self.model(inputs)
                        orig_probs = F.softmax(orig_logits, dim=1)
                        orig_preds = torch.argmax(orig_probs, dim=1)

                    if self.dfr_type == "dropout":
                        self.model.train()
                        logits = self.model(inputs)
                    elif self.dfr_type == "earlystop":
                        logits = self.earlystop_model(inputs)
                    elif self.dfr_type == "earlystop_dropout":
                        self.earlystop_model.train()
                        logits = self.earlystop_model(inputs)
                    else:
                        # Dummy prediction for misclassification.
                        logits = self.model(inputs)
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    all_orig_logits.append(orig_logits)
                    all_logits.append(logits)
                    all_orig_probs.append(orig_probs)
                    all_probs.append(probs)
                    all_targets.append(targets)

            all_orig_logits = torch.cat(all_orig_logits)
            all_logits = torch.cat(all_logits)
            all_orig_probs = torch.cat(all_orig_probs)
            all_probs = torch.cat(all_probs)
            all_targets = torch.cat(all_targets)
            kldiv = torch.mean(F.kl_div(torch.log(all_probs), all_orig_probs, reduction="none"), dim=1).squeeze()
            # should class_weights be used here?
            loss = F.cross_entropy(all_orig_logits, all_targets, reduction="none").squeeze()

            #isect = torch.argmax(all_orig_probs, dim=1) != all_targets
            #isect_kldiv = kldiv[isect]
            #isect_inds = all_inds[to_np(isect)]

            all_targets = to_np(all_targets)

            del all_orig_logits
            del all_logits
            del all_orig_probs
            del all_probs

            # KL Histogram
            """
            DATASET = "MultiNLI"
            NUM = 100
            minority = {
                "Waterbirds": [2,4],
                "CelebA": [4],
                "FMOW": [3,5],
                "MultiNLI": [2,4,6],
            }
            import matplotlib.pyplot as plt
            import sys
            plt.rcParams["figure.figsize"] = (8, 5)
            #top_kl = np.sort(to_np(kldiv))[-NUM:]
            #top_kl = np.sort(to_np(isect_kldiv))[-NUM:]
            top_kl = np.sort(to_np(loss))[-NUM:]
            #top_kl_inds = np.argsort(to_np(kldiv))[-NUM:]
            top_kl_inds = np.argsort(to_np(loss))[-NUM:]
            #top_kl_inds = np.argsort(to_np(isect_kldiv))[-NUM:]
            #y = [[] for _ in range(len(new_set.groups[1:]))]
            y = [[], []]
            for i, k in zip(top_kl_inds, top_kl):
                q = all_inds[i]
                #q = isect_inds[i]
                for j, g in enumerate(new_set.groups[1:]):
                    if q in g:
                        if j+1 in minority[DATASET]:
                            y[1].append(k)
                        else:
                            y[0].append(k)
                        break
            #label = [f"Group {j+1}" for j in range(len(new_set.groups[1:]))]
            label = ["Majority", "Minority"]
            plt.hist(y, "auto", stacked=True, density=True, label=label)
            plt.legend()
            plt.title(f"Distribution of top {NUM} early stop misclassifications {DATASET}, Seed 1")
            #plt.xlabel("KL Divergence")
            plt.xlabel("CE Loss")
            plt.savefig("hist.png")
            print([len(r) for r in y])
            sys.exit(0)
            """

            top_num = int(self.num_data * self.gamma)
            if self.gamma == 1:
                bottom_num = 0
                agreements = []
            else:
                bottom_num = int(self.num_data * (1 - self.gamma))

            if "dropout" in self.dfr_type or self.dfr_type == "earlystop":
                if not self.kl_ablation:
                    #disagreements = to_np(torch.topk(kldiv, k=int(ceil(len(kldiv)*self.num_data // 2)))[1])
                    #agreements = to_np(torch.topk(-kldiv, k=int(ceil(len(kldiv)*self.num_data // 2)))[1])
                    disagreements = to_np(torch.topk(kldiv, k=top_num)[1])
                    if bottom_num:
                        agreements = to_np(torch.topk(-kldiv, k=bottom_num)[1])
                    """
                    if random_num:
                        agreements = np.random.choice(
                            np.delete(np.arange(len(kldiv)), disagreements),
                            size=random_num,
                            replace=False,
                        )
                    """
            elif "miscls" in self.dfr_type:
                #disagreements = to_np(torch.topk(loss, k=int(ceil(len(kldiv)*self.num_data // 2)))[1])
                #agreements = to_np(torch.topk(-loss, k=int(ceil(len(kldiv)*self.num_data // 2)))[1])
                disagreements = to_np(torch.topk(loss, k=top_num)[1])
                if bottom_num:
                    agreements = to_np(torch.topk(-loss, k=bottom_num)[1])
                """
                if random_num:
                    agreements = np.random.choice(
                        np.delete(np.arange(len(loss)), disagreements),
                        size=random_num,
                        replace=False,
                    )
                """

            disagree = all_inds[disagreements].tolist()
            disagree_targets = all_targets[disagreements].tolist()
            if len(agreements) > 0:
                agree = all_inds[agreements].tolist()
                agree_targets = all_targets[agreements].tolist()
                indices = np.concatenate((disagree, agree))
            else:
                agree = []
                agree_targets = []
                indices = np.asarray(disagree)

        # Uses disagreement set as new training set for DFR.
        new_set.train_indices = new_set.val_indices
        self.dataset_train = Subset(new_set, indices)

        # Prints number of data in each group.
        self.print_disagreements_by_group(new_set, all_inds, disagree, agree, indices=indices)

    def setup(self, stage=None):
        """Instantiates and preprocesses datasets.

        Performs disagreement if self.model (i.e., the model which calculates
        disagreements) is specified.
        
        Args:
            stage: The stage of training; either "fit", "test", or None (both).
        """

        if stage == "fit" or stage is None:
            dataset_train = self.dataset_class(
                self.data_dir,
                train=True,
                transform=self.train_transforms,
            )

            dataset_val = self.dataset_class(
                self.data_dir,
                train=True,
                transform=self.val_transforms,
            )

            # Creates disagreement sets in addition to regular train/val split.
            dataset_train, dataset_val = self.train_preprocess(dataset_train, dataset_val)
            self.dataset_train = self._split_dataset(dataset_train, disagreement_proportion=self.disagreement_proportion)
            self.dataset_val, self.dataset_disagreement = self._split_dataset(
                dataset_val,
                disagreement_proportion=self.disagreement_proportion,
                train=False,
            )

            # Performs disagreement and sets new train dataset.
            if self.model:
                print("Computing disagreements...")
                self.disagreement()
                del self.model

        if stage == "test" or stage is None:
            dataset_test = self.dataset_class(
                self.data_dir,
                train=False,
                transform=self.test_transforms,
            )
            self.dataset_test = self.test_preprocess(dataset_test)

