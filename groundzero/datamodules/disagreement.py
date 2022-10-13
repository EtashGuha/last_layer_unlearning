"""DataModule for disagreement-based datasets."""

# Imports Python builtins.
import random

# Imports Python packages.
import numpy as np

# Imports PyTorch packages.
import torch
import torch.nn.functional as F

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
        disagreement_set: Either 'train' or 'val'; the dataset for disagreement.
        disagreement_proportion: The proportion of the dataset to use for disagreement.
        gamma: The proportion of agreements with which to augment disagreements.
        orig_dfr: Whether to use group labels to perform original DFR.
        misclassification_dfr: Whether to perform misclassification DFR.
        dropout_dfr: Whether to perform dropout DFR.
        rebalancing: Whether to class-rebalance the disagreement set.
        disagreement_ablation: Whether to only use agreement points.
        dataset_disagreement: A groundzero.datamodules.Dataset for disagreement.
    """

    def __init__(
        self,
        args,
        *xargs,
        model=None,
        gamma=1,
        orig_dfr=False,
        misclassification_dfr=False,
        dropout_dfr=False,
        rebalancing=False,
        disagreement_ablation=False,
    ):
        """Initializes a Disagreement DataModule.
        
        Args:
            args:
            model: The groundzero.models.Model used for disagreement.
            gamma: The proportion of agreements with which to augment disagreements.
            orig_dfr: Whether to use group labels to perform original DFR.
            misclassification_dfr: Whether to perform misclassification DFR.
            dropout_dfr: Whether to perform dropout DFR.
            rebalancing: Whether to class-rebalance the disagreement set.
            disagreement_ablation: Whether to only use agreement points.
        """

        super().__init__(args, *xargs)
 
        self.model = model.cuda() if model else None
        self.disagreement_set = args.disagreement_set
        self.disagreement_proportion = args.disagreement_proportion
        self.gamma = gamma
        self.orig_dfr = orig_dfr
        self.misclassification_dfr = misclassification_dfr
        self.dropout_dfr = dropout_dfr
        self.rebalancing = rebalancing
        self.disagreement_ablation = disagreement_ablation

    def load_msg(self):
        """Returns a descriptive message about the DataModule configuration."""

        msg = super().load_msg()
        msg = msg[:-1] + (
            f", with disagreement set {self.disagreement_set}"
            f" and proportion {self.disagreement_proportion}."
        )
        return msg

    def _split_dataset(self, dataset, disagreement_proportion=None, train=True):
        """Splits dataset into training and validation subsets.

        If disagreement_proportion is specified, uses that proportion of the
        dataset for disagreement and the rest as normal. For example, one could
        use half the val dataset for disagreement and the remaining half for
        validation. The exception is if disagreement_proportion == 1; in this
        case, the entire set will be used for disagreement and as usual. This
        is useful for doing disagreement on the train dataset, where we want
        to use the whole dataset for training and then check our disagreements
        against a resource-constrained model.
        
        Args:
            dataset: A groundzero.datamodules.Dataset.
            disagreement_proportion: The proportion of the dataset for disagreement.
            train: Whether to return the train set or val set.

        Returns:
            A torch.utils.data.Subset of the given dataset with the desired split.
        """

        inds = dataset.train_indices if train else dataset.val_indices

        if disagreement_proportion:
            if float(disagreement_proportion) != 1.:
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
                return Subset(dataset, inds), Subset(dataset, inds)
        else:
            return Subset(dataset, inds)

    def disagreement_dataloader(self):
        """Returns DataLoader for the disagreement set."""

        return self._data_loader(self.dataset_disagreement)

    def rebalance_groups(self, indices, dataset):
        """Uses group labels to balance dataset across groups.
        
        Args:
            indices: An np.ndarray of data indices.
            dataset: A groundzero.datamodules.Dataset.
            
        Returns:
            An np.ndarray of indices with equal data from each group.
        """

        # Gets indices corresponding to each group. Does not get indices
        # for group 0, which is by convention the group of all indices.
        group_inds = []
        for group in dataset.groups[1:]:
            group_inds.append(np.intersect1d(indices, group))

        for g in group_inds:
            random.shuffle(g)

        # Truncates each group to have the same number of indices.
        m = min([len(g) for g in group_inds])
        indices = np.concatenate([g[:m] for g in group_inds])

        return indices

    def rebalance_classes(self, indices, targets):
        """Uses class labels to balance dataset across classes.

        Args:
            indices: An np.ndarray of data indices.
            targets: An np.ndarray of class targets.

        Returns:
            An np.ndarray of indices with equal data from each class.
        """

        # TODO: Extend to the case when targets are not consecutive ints.
        # TODO: Generalize to any number of classes.

        # Skips balancing if the list is empty (e.g., if gamma == 0).
        if len(targets) == 0:
            return indices

        # Computes the class imbalance of the data.
        num_one = np.count_nonzero(targets)
        num_zero = len(targets) - num_one
        to_remove = abs(num_zero - num_one)
        if to_remove == 0:
            return indices

        # Sets locations of minority class to True.
        mask = targets == 1 if num_zero > num_one else targets == 0

        # Removes majority class data until they equal the minority class.
        false_inds = (~mask).nonzero()[0]
        keep = np.random.choice(
            false_inds,
            len(false_inds) - to_remove,
            replace=False,
        )
        mask[keep] = True
        
        return indices[mask]

    def print_disagreements_by_group(self, dataset, all_inds, disagree, agree, indices=None):
        """Prints number of disagreements or agreements occuring in each group.
        
        Args:
            dataset: A groundzero.datamodules.Dataset.
            all_inds: An np.ndarray of all indices in the disagreement set.
            disagree: An np.ndarray of all disagreed indices.
            agree: An np.ndarray of all agreed indices.
        """

        if indices:
            labels_and_inds = zip(
                ("All", "Disagreements", "Agreements", "Final Indices"),
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
        
        self.dataset_disagreement is initially the specified self.disagreement_proportion
        of self.disagreement_set. Here, we perform some computation (i.e., the
        actual disagreement) on self.dataset_disagreement to get the indices for
        DFR. Then, we set these indices as self.dataset_train for DFR training.
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

        # Gets all the relevant indices from the disagreement set.
        if self.disagreement_set == "train":
            all_inds = dataloader.dataset.train_indices
        elif self.disagreement_set == "val":
            all_inds = dataloader.dataset.val_indices

        if self.orig_dfr:
            # Gets all indices and targets from the disagreement set.
            targets = []
            for batch in dataloader:
                targets.extend(to_np(batch[1]))
            targets = np.asarray(targets)
            indices = all_inds

            # Performs group balancing on the disagreements for original DFR.
            if self.rebalancing:
                indices = self.rebalance_groups(indices, new_set)
        else:
            all_orig_probs = []
            all_probs = []
            all_targets = []

            # Performs misclassifiation or dropout disagreements with self.model.
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    inputs, targets = batch
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                    if self.misclassification_dfr:
                        # Gets predictions from non-dropout model.
                        self.model.eval()
                        logits = self.model(inputs)
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)

                        # Gets misclassifications.
                        disagreements = to_np(torch.logical_xor(preds, targets))
                    else:
                        # Gets predictions from non-dropout model.
                        self.model.eval()
                        orig_logits = self.model(inputs)
                        orig_probs = F.softmax(orig_logits, dim=1)
                        orig_preds = torch.argmax(orig_probs, dim=1)

                        # Gets predictions from dropout model.
                        self.model.train()
                        logits = self.model(inputs)
                        probs = F.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)

                        """
                        all_orig_probs.append(orig_probs)
                        all_probs.append(probs)
                        all_targets.extend(to_np(targets))
                        """

                        # Gets dropout disagreements.
                        disagreements = to_np(torch.logical_xor(preds, orig_preds))

                    #if self.misclassification_dfr:
                    inds = all_inds[(i * batch_size):min(((i+1) * batch_size), len(all_inds))]
                    disagree.extend(inds[disagreements].tolist())
                    disagree_targets.extend(targets[disagreements].tolist())
                    agree.extend(inds[~disagreements].tolist())
                    agree_targets.extend(targets[~disagreements].tolist())
            
            """
            if self.dropout:
                all_orig_probs = torch.cat(all_orig_probs)
                all_probs = torch.cat(all_probs)
                kldiv = torch.mean(F.kl_div(torch.log(all_probs), all_orig_probs, reduction="none"), dim=1).squeeze()
                print(kldiv.shape)

                del all_orig_probs
                del all_probs

                disagreements = to_np(torch.topk(kldiv, k=60)[1])
                print(disagreements.shape)
                agreements = to_np(torch.topk(-kldiv, k=60)[1])
                all_targets = np.asarray(all_targets)

                disagree = all_inds[disagreements].tolist()
                disagree_targets = all_targets[disagreements].tolist()
                agree = all_inds[agreements].tolist()
                agree_targets = all_targets[agreements].tolist()
            """

            if self.gamma > 0:
                # Gets a gamma proportion of agreement points.
                num_agree = int(self.gamma * len(disagree))
                c = list(zip(agree, agree_targets))
                random.shuffle(c)
                agree, agree_targets = zip(*c)
                agree = agree[:num_agree]
                agree_targets = agree_targets[:num_agree]

                if self.disagreement_ablation:
                    # Ablates disagreement set (just uses agreement points).
                    # Note that we still get a gamma proportion of
                    # agreement points first, to fix the number of data.
                    disagree = []
                    disagree_targets = []
            elif self.gamma == 0:
                # Ablates agreement set (just uses disagreement points).
                agree = []
                agree_targets = []
            else:
                raise ValueError("Gamma must be non-negative.")
                
            # Converts all lists to np.ndarrays.
            disagree = np.asarray(disagree, dtype=np.int64)
            disagree_targets = np.asarray(disagree_targets, dtype=np.int64)
            agree = np.asarray(agree, dtype=np.int64)
            agree_targets = np.asarray(agree_targets, dtype=np.int64)

            if self.rebalancing:
                # Prints number of data in each group prior to balancing.
                self.print_disagreements_by_group(new_set, all_inds, disagree, agree)

                # Performs class balancing on the disagreements. Note that the
                # disagreements and agreements are balanced separately.
                disagree = self.rebalance_classes(disagree, disagree_targets)
                agree = self.rebalance_classes(agree, agree_targets)
             
            # Adds disagreement and agreement points to indices.
            indices = np.concatenate((disagree, agree))

        # Uses disagreement set as new training set for DFR.
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
            self.dataset_train = self._split_dataset(dataset_train)
            if self.disagreement_set == "train":
                _, self.dataset_disagreement = self._split_dataset(
                    dataset_val,
                    disagreement_proportion=self.disagreement_proportion,
                )
                self.dataset_val = self._split_dataset(dataset_val, train=False)
            elif self.disagreement_set == "val":
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

