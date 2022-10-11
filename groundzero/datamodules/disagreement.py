"""DataModule for disagreement-based datasets."""

# Imports Python builtins.
import random

# Imports Python packages.
import numpy as np

# Imports PyTorch packages.
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

# Imports groundzero packages.
from groundzero.datamodules.datamodule import DataModule
from groundzero.utils import to_np

# use multiple inheritance for WaterbirdsDisagreement (e.g., from Waterbirds and from Disagreement)?

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
        dataset_disagreement: A groundzero.datamodules.Dataset for disagreement.
    """

    def __init__(
        self,
        *args,
        model=None,
        gamma=1,
        orig_dfr=False,
        misclassification_dfr=False,
        dropout_dfr=False,
        rebalancing=False,
    ):
        """Initializes a Disagreement DataModule.
        
        Args:
            model: The groundzero.models.Model used for disagreement.
            gamma: The proportion of agreements with which to augment disagreements.
            orig_dfr: Whether to use group labels to perform original DFR.
            misclassification_dfr: Whether to perform misclassification DFR.
            dropout_dfr: Whether to perform dropout DFR.
            rebalancing: Whether to class-rebalance the disagreement set.
        """

        super().__init__(*args)
 
        self.model = self.model.cuda() if model else None
        self.disagreement_set = args.disagreement_set
        self.disagreement_proportion = args.disagreement_proportion
        self.gamma = gamma
        self.misclassification_dfr = misclassification_dfr
        self.full_set_dfr = full_set_dfr
        self.dropout = dropout
        self.rebalancing = rebalancing

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
        num_one = np.count_nonzero(t)
        num_zero = len(t) - num_one
        to_remove = abs(num_zero - num_one)
        if to_remove == 0:
            return indices

        # Sets locations of minority class to True.
        if num_zero > num_one:
            mask = t == 1
        else:
            mask = t == 0

        # Removes majority class data until they equal the minority class.
        false_inds = (~mask).nonzero()[0]
        keep = np.random.choice(
            false_inds,
            len(false_inds) - to_remove,
            replace=False,
        )
        mask[keep] = True
        
        return indices[mask]

    def disagreement(self):
        dataloader = self.disagreement_dataloader()

        new_set = self.dataset_class(
            self.data_dir,
            train=True,
            transform=self.train_transforms,
        )

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
            all_orig_probs = []
            all_probs = []
            all_targets = []
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
                        #all_orig_probs.append(orig_probs)
                        #all_probs.append(probs)
                        #all_targets.extend(to_np(targets))
                        disagreements = to_np(torch.logical_xor(preds, orig_preds))
                    else:
                        raise ValueError("Can't do disagreement w/o dropout")

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

                disagree = self.rebalance_classes(disagree, disagree_targets)
                agree = self.rebalance_classes(agree, agree_targets)
                            
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

