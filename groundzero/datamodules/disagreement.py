"""DataModule for disagreement-based datasets."""

# Imports Python builtins.
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

# TODO: Change disagreement names to SELF or finetuning?


class Disagreement(DataModule):
    """DataModule for disagreement sets used for last-layer retraining.

    The original DFR procedure uses group annotations to construct a reweighting
    dataset that has equal data from every group. We propose using disagreement
    between the ERM model and a regularized (e.g., dropout or early-stopped) 
    model as an alternative. This enables construction of nearly-group-balanced
    reweighting datasets without the need for group annotations.

    This class is currently only defined for datasets with a preset val split,
    i.e., groundzero.datasets which have train_indices and val_indices.

    Attributes:
    """

    def __init__(
        self,
        args,
        *xargs,
        early_stop_model=None,
        model=None,
        num_data=None,
        worst_group_pct=None,
    ):
        """Initializes a Disagreement DataModule.
        
        Args:
            args:
            *xargs:
            model: The groundzero.models.Model used for disagreement.
        """

        super().__init__(args, *xargs)
        
        self.heldout_pct = 0.5

        self.balance_finetune = args.balance_finetune
        self.finetune_type = args.finetune_type if hasattr(args, "finetune_type") else None
        self.split = args.split
        self.train_pct = args.train_pct

        self.early_stop_model = early_stop_model.cuda() if early_stop_model else None
        self.model = model.cuda() if model else None
        self.num_data = num_data
        self.worst_group_pct = worst_group_pct

    def load_msg(self):
        """Returns a descriptive message about the DataModule configuration."""

        msg = super().load_msg()
        msg = msg[:-1] + (
            f", with proportion {self.heldout_pct}."
        )
        return msg

    def _split_dataset(self, dataset_train, dataset_val):
        """Splits dataset into training and validation subsets.

        Args:
            dataset: A groundzero.datamodules.Dataset.

        Returns:
        """

        train_inds = dataset_train.train_indices
        val_inds = dataset_val.val_indices

        random.shuffle(val_inds)

        disagreement_num = int(self.heldout_pct * len(val_inds))
        new_val_inds = val_inds[disagreement_num:]

        if self.split == "train" and self.train_pct == 100:
            new_train_inds = train_inds
            new_dis_inds = val_inds[:disagreement_num]
        elif self.split == "train":
            random.shuffle(train_inds)
            train_num = int(len(train_inds) * self.train_pct / 100)
            new_train_inds = train_inds[:train_num]
            new_dis_inds = train_inds[train_num:]
        elif self.split == "combined":
            combined_inds = np.concatenate((train_inds, val_inds))
            train_num = int((len(train_inds) + disagreement_num) * self.train_pct / 100)
            new_dis_num = len(combined_inds) - int((1 - self.heldout_pct) * len(val_inds))
            new_combined_inds = combined_inds[:new_dis_num]

            random.shuffle(new_combined_inds)
            new_train_inds = new_combined_inds[:train_num]
            new_dis_inds = new_combined_inds[train_num:]

        dataset_t = Subset(dataset_train, new_train_inds)
        dataset_d = Subset(dataset_val, new_dis_inds)
        dataset_v = Subset(dataset_val, new_val_inds)

        # TODO: Is this necessary?
        if self.split == "combined":
            dataset_t.train_indices = list(range(len(dataset_t)))

        dataset_d.val_indices = new_dis_inds
        dataset_v.val_indices = new_val_inds

        return dataset_t, dataset_d, dataset_v

    def train_dataloader(self):
        """Returns DataLoader for the train dataset (after disagreement).
        
           This method also handles class- and group-balancing, either by
           sampling the data to ensure balanced minibatches ("sampler")
           or by training on a balanced subset of the data ("subset").
           TODO: Describe worst-group-pct ablation.
        """
        
        if self.finetune_type == "group-unbalanced retraining":
            if self.worst_group_pct:
                # Sets the worst groups (zero-indexed).
                min_group_nums = {
                    "Waterbirds": np.array([1, 2]),
                    "CelebA": np.array([3]),
                    "CivilComments": np.array([3]),
                    "MultiNLI": np.array([3, 5]),
                }

                maj_group_nums = {
                    "Waterbirds": np.array([0, 3]),
                    "CelebA": np.array([2]),
                    "CivilComments": np.array([2]),
                    "MultiNLI": np.array([2, 4]),
                }

                dataset = self.__class__.__bases__[0].__name__[:-12]
                min_groups = min_group_nums[dataset]
                maj_groups = maj_group_nums[dataset]
                totals = np.array([len(x) for x in self.dataset_train.groups[1:]])
                num_groups = len(totals)
                num_maj_groups = len(maj_groups)

                smallest_min_group = min([t for j, t in enumerate(totals)
                                          if j in min_groups])
                smallest_maj_group = min([t for j, t in enumerate(totals)
                                          if j in maj_groups])

                downsample_num = min(smallest_maj_group // 2, smallest_min_group)
                totals = [downsample_num] * num_groups

                """
                # Makes CelebA and CivilComments class-unbalanced.
                maj_ratio = sum(totals[:2]) / sum(totals)
                maj = int(-(maj_ratio * downsample_num) / (maj_ratio - 1))
                totals = [maj, maj, downsample_num, downsample_num]
                """

                removed = 0
                for j, _ in enumerate(totals):
                    if j in min_groups:
                        new = int(totals[j] * self.worst_group_pct / 100)
                        removed += totals[j] - new
                        totals[j] = new

                maj_per_group = removed // num_maj_groups
                for j, t in enumerate(totals):
                    if j in maj_groups:
                        totals[j] += maj_per_group

                indices = self.dataset_train.train_indices
                random.shuffle(indices)

                new_indices = []
                nums = [0] * num_groups
                for x in indices:
                    for j, group in enumerate(self.dataset_train.groups[1:]):
                        if x in group and nums[j] < totals[j]:
                            new_indices.append(x) 
                            nums[j] += 1
                            break
                print(nums)

                new_indices = np.array(new_indices)
                self.dataset_train = Subset(self.dataset_train, new_indices)

                self.balanced_sampler = False
                return super().train_dataloader()
            elif self.balance_finetune == "sampler":
                self.balanced_sampler = True
                return super().train_dataloader()
            elif self.balance_finetune == "subset":
                raise NotImplementedError()
            elif self.balance_finetune == "none":
                self.balanced_sampler = False
                return super().train_dataloader()
        elif self.finetune_type == "group-balanced retraining":
            indices = self.dataset_train.train_indices
            groups = np.zeros(len(indices), dtype=np.int32)
            for i, x in enumerate(indices):
                for j, group in enumerate(self.dataset_train.groups[1:]):
                    if x in group:
                        groups[i] = j

            if self.balance_finetune == "sampler":
                counts = np.bincount(groups)
                label_weights = 1. / counts
                weights = label_weights[groups]
                sampler = WeightedRandomSampler(weights, len(weights))
                return self._data_loader(self.dataset_train, sampler=sampler)
            elif self.balance_finetune == "subset":
                # TODO: Shuffle indices and groups together?
                min_group = min([len(x) for x in self.dataset_train.groups[1:]])
                counts = [0] * len(self.dataset_train.groups[1:])

                subset = []
                for idx, group in zip(indices, groups):
                    if counts[group] < min_group:
                        subset.append(idx)
                        counts[group] += 1
                print(f"Balanced subset: {counts}")

                self.dataset_train = Subset(self.dataset_train, subset)

                self.balanced_sampler = False
                return super().train_dataloader()
            elif self.balance_finetune == "none":
                raise ValueError("Set balance_finetune for group-balanced retraining.")
        else:
            if self.balance_finetune == "sampler":
                self.balanced_sampler = True
                return super().train_dataloader()
            elif self.balance_finetune == "subset":
                raise NotImplementedError()
            elif self.balance_finetune == "none":
                self.balanced_sampler = False
                return super().train_dataloader()

    def val_dataloader(self):
        dataloaders = super().val_dataloader()
        return dataloaders

    def test_dataloader(self):
        dataloaders = super().test_dataloader()
        return dataloaders

    def disagreement_dataloader(self):
        """Returns DataLoader for the disagreement set."""

        return self._data_loader(self.dataset_disagreement)

    def print_disagreements_by_group(self, dataset, all_inds, disagree=None):
        """Prints number of disagreements occuring in each group.
        
        Args:
            dataset: A groundzero.datamodules.Dataset.
            all_inds: An np.ndarray of all indices in the disagreement set.
            disagree: An optional np.ndarray of all disagreed indices.
        """

        labels_and_inds = zip(
            ("All", "Disagreements"),
            (all_inds, disagree),
        )

        print("Disagreements by group")
        for label, inds in labels_and_inds:
            if inds is not None:
                nums = []
                for group in dataset.groups[1:]: # skips group 0 (all inds)
                    nums.append(len(np.intersect1d(inds, group)))
                print(f"{label}: {nums}")
    
    def disagreement(self):
        """Computes finetuning set and saves it as self.dataset_train.
        
        self.dataset_disagreement is initially the self.heldout_pct of the
        held-out set. Here, we perform some computation (i.e., the actual
        disagreement) on self.dataset_disagreement to get indices for
        finetuning. Then, we set these indices as self.dataset_train.
        """

        dataloader = self.disagreement_dataloader()
        batch_size = dataloader.batch_size

        all_inds = dataloader.dataset.val_indices

        new_set = self.dataset_class(
            self.data_dir,
            train=True,
            transform=self.train_transforms,
        )
        new_set.train_indices = np.concatenate((new_set.train_indices, new_set.val_indices))

        if "retraining" in self.finetune_type:
            self.dataset_train = Subset(new_set, all_inds)
            self.print_disagreements_by_group(new_set, all_inds)
            return

        all_orig_logits = []
        all_logits = []
        all_orig_probs = []
        all_probs = []
        all_targets = []

        # Performs misclassification or disagreement with self.model.
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                inputs, targets = batch
                inputs = inputs.cuda()
                targets = targets.cuda()

                if "early-stop" in self.finetune_type:
                    self.early_stop_model.eval()
                    orig_logits = self.early_stop_model(inputs)
                    orig_probs = F.softmax(orig_logits, dim=1)
                    orig_preds = torch.argmax(orig_probs, dim=1)
                else:
                    self.model.eval()
                    orig_logits = self.model(inputs)
                    orig_probs = F.softmax(orig_logits, dim=1)
                    orig_preds = torch.argmax(orig_probs, dim=1)

                if "misclassification" in self.finetune_type:
                    logits = self.model(inputs)
                else:
                    if "dropout" in self.finetune_type:
                        self.model.train()
                        logits = self.model(inputs)
                    elif "early-stop" in self.finetune_type:
                        logits = self.early_stop_model(inputs)

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
        
        kldiv = F.kl_div(torch.log(all_probs), all_orig_probs, reduction="none")
        kldiv = torch.mean(kldiv, dim=1).squeeze()
        loss = F.cross_entropy(all_orig_logits, all_targets, reduction="none").squeeze()

        if "misclassification" in self.finetune_type:
            disagreements = to_np(torch.topk(loss, k=self.num_data)[1])
        elif "disagreement" in self.finetune_type:
            disagreements = to_np(torch.topk(kldiv, k=self.num_data)[1])

        disagree = all_inds[disagreements]
            
        self.dataset_train = Subset(new_set, disagree)
        self.print_disagreements_by_group(new_set, all_inds, disagree=disagree)
        
    def setup(self, stage=None):
        """Instantiates and preprocesses datasets.

        Performs disagreement if self.model (i.e., the model which calculates
        disagreements) is specified.
        
        Args:
            stage: The stage of training; either "fit", "test", or None (both).
        """

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

        dataset_test = self.dataset_class(
            self.data_dir,
            train=False,
            transform=self.test_transforms,
        )

        # Creates disagreement sets in addition to regular train/val split.
        dataset_train, dataset_val = self.train_preprocess(dataset_train, dataset_val)
        self.dataset_train, self.dataset_disagreement, self.dataset_val = \
            self._split_dataset(dataset_train, dataset_val)

        dataset_test = self.test_preprocess(dataset_test)
        self.dataset_test = dataset_test
        
        if stage == "fit" or stage is None:
            # Performs disagreement and sets new train dataset.
            if self.model:
                print("Computing disagreements...")
                self.disagreement()
                del self.model

