"""Parent class for a vision classification datamodule."""

# Imports Python builtins.
from abc import abstractmethod
import random

# Imports Python packages.
import numpy as np

# Imports PyTorch packages.
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from pl_bolts.datamodules.vision_datamodule import VisionDataModule


class DataModule(VisionDataModule):
    """Parent class for a vision classification datamodule.

    Extends the basic PL.VisionDataModule to play nice with
    torchvision.datasets.VisionDataset and adds some custom functionality
    such as label noise, balanced sampling, etc.

    Attributes:
        dataset_class: A torchvision.datasets.VisionDataset class.
        num_classes: The number of classes.
        balanced_sampler: Whether to use a class-balanced random sampler during training.
        data_augmentation: Whether to use data augmentation during training.
        label_noise: Whether to add label noise during training.
        dataset_train: A torchvision.datasets.VisionDataset for training.
        dataset_val: A torchvision.datasets.VisionDataset for validation.
        dataset_test: A torchvision.datasets.VisionDataset for testing.
        train_transforms: Composition of torchvision.transforms for the train set.
        val_transforms: Composition of torchvision.transforms for the val set.
        test_transforms: Composition of torchvision.transforms for the test set.
    """

    def __init__(self, args, dataset_class, num_classes):
        """Initializes a DataModule and sets transforms.

        Args:
            args: The configuration dictionary.
            dataset_class: A class which inherits from torchvision.datasets.VisionDataset.
            num_classes: The number of classes.
        """

        super().__init__(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            drop_last=False,
            normalize=True,
            num_workers=args.num_workers,
            pin_memory=True,
            seed=args.seed,
            shuffle=True,
            val_split=args.val_split,
        )

        self.dataset_class = dataset_class
        self.num_classes = num_classes
        
        self.balanced_sampler = args.balanced_sampler
        self.data_augmentation = args.data_augmentation
        self.label_noise = args.label_noise
         
        self.train_transforms = self.default_transforms()
        self.val_transforms = self.default_transforms()
        self.test_transforms = self.default_transforms()

        if self.data_augmentation:
            self.train_transforms = self.augmented_transforms()

    @abstractmethod
    def augmented_transforms(self):
        """Returns torchvision.transforms for use with data augmentation."""

    @abstractmethod
    def default_transforms(self):
        """Returns default torchvision.transforms."""
 
    def prepare_data(self):
        """Downloads datasets to disk if necessary."""

        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def load_msg(self):
        """Returns a descriptive message about the DataModule configuration."""

        msg = f"Loading {type(self).__name__}"
 
        # TODO: I don't think this works as intended.
        if hasattr(self, "dataset_val") and self.dataset_val.val_indices:
            msg = msg + " with preset val split."
        elif hasattr(self, "dataset_val"):
            msg = msg + f" with {int(self.val_split * 100)}% random val split."
        else:
            msg = msg + " with test split."

        if self.data_augmentation:
            msg = msg[:-1] + " and data augmentation."
        if self.label_noise:
            msg = msg[:-1] + f" and {int(self.label_noise * 100)}% label noise."

        return msg

    def train_preprocess(self, dataset_train, dataset_val):
        """Preprocesses train and val datasets. Here, injects label noise.

        Args:
            dataset_train: A torchvision.datasets.VisionDataset for training.
            dataset_val: A torchvision.datasets.VisionDataset for validation.

        Returns:
            The modified train and val datasets.
        """

        if self.label_noise:
            # Adds label noise to train dataset (i.e., randomly selects a new
            # label for the specified proportion of training datapoints).
            # TODO: Ensure train_indices, etc. are specified for preset splits.
            # If label noise is nonzero with a preset split, but train_indices
            # is not set, then this procedure will apply noise to the val set!
            if hasattr(dataset_train, "train_indices"):
                train_indices = dataset_train.train_indices
            else:
                train_indices = randperm(
                    len(dataset_train),
                    generator=torch.Generator().manual_seed(self.seed),
                ).tolist()
                train_length = self._get_splits(len(dataset_train))[0]
                train_indices = train_indices[:train_length]

            num_labels = len(train_indices)
            num_noised_labels = int(self.label_noise * num_labels)

            for i, target in enumerate(train_indices[:num_noised_labels]):
                labels = [j for j in range(self.num_classes) if j != i]
                dataset_train.targets[i] = random.choice(labels)

        return dataset_train, dataset_val

    def test_preprocess(self, dataset_test):
        """Preprocesses test dataset. Does nothing here, but can be overidden.

        Args:
            dataset_test: A torchvision.datasets.VisionDataset for testing.

        Returns:
            The modified test dataset.
        """

        return dataset_test

    def setup(self, stage=None):
        """Instantiates and preprocesses datasets.
        
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

            dataset_train, dataset_val = self.train_preprocess(dataset_train, dataset_val)
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)
            
        if stage == "test" or stage is None:
            dataset_test = self.dataset_class(
                self.data_dir,
                train=False,
                transform=self.test_transforms,
            )

            self.dataset_test = self.test_preprocess(dataset_test)

    def _split_dataset(self, dataset, train=True):
        """Splits dataset into training and validation subsets.
        
        Args:
            dataset: A torchvision.datasets.VisionDataset.
            train: Whether to return the train set or val set.

        Returns:
            A torch.utils.data.Subset of the given dataset with the desired split.
        """

        if train and dataset.train_indices and dataset.val_indices:
            # Calculates a preset split based on the given indices.
            dataset_train = Subset(dataset, dataset.train_indices)
            dataset_val = Subset(dataset, dataset.val_indices)
        else:
            # Calculates a random split based on args.val_split.
            len_dataset = len(dataset)
            splits = self._get_splits(len_dataset)
            dataset_train, dataset_val = random_split(
                dataset,
                splits,
                generator=torch.Generator().manual_seed(self.seed),
            )

        if train:
            return dataset_train
        return dataset_val

    def train_dataloader(self):
        """Returns DataLoader for the train dataset."""

        if self.balanced_sampler:
            # Instantiates balanced sampler if desired.
            # TODO: Change for if labels are not (0, ..., num_classes).
            #new_set = self.dataset_class(self.data_dir, train=True)
            #subset_indices = new_set.train_indices
            #targets = new_set.targets[subset_indices]
            indices = self.dataset_train.train_indices
            targets = self.dataset_train.targets[indices]

            counts = np.bincount(targets)
            label_weights = 1. / counts
            weights = label_weights[targets]
            sampler = WeightedRandomSampler(weights, len(weights))

            return self._data_loader(self.dataset_train, sampler=sampler)

        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self):
        """Returns DataLoader(s) for the val dataset."""

        if len(self.dataset_val.groups):
            # Returns a list of DataLoaders for each group/split.
            # TODO: Breaks if self.dataset_val is a Subset?
            dataloaders = []
            for group in range(len(self.dataset_val.groups)):
                dataloaders.append(
                    self._data_loader(
                        self._split_dataset(
                            self.dataset_class(
                                self.data_dir,
                                train=True,
                                transform=self.default_transforms(),
                                group=group,
                            ),
                            train=False,
                )))

            return dataloaders
        return self._data_loader(self.dataset_val)

    def test_dataloader(self):
        """Returns DataLoader(s) for the test dataset."""

        if len(self.dataset_test.groups):
            # Returns a list of DataLoaders for each group/split.
            # TODO: Breaks if self.dataset_test is a Subset?
            dataloaders = []
            for group in range(len(self.dataset_test.groups)):
                dataloaders.append(
                    self._data_loader(
                        self.dataset_class(
                            self.data_dir,
                            train=False,
                            transform=self.default_transforms(),
                            group=group,
                )))

            return dataloaders
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset, shuffle=False, sampler=None):
        """Instantiates DataLoader with the given dataset.

        Args:
            dataset: A torchvision.datasets.VisionDataset.
            shuffle: Whether to shuffle the data indices.
            sampler: A torch.utils.data.Sampler for selecting data indices.

        Returns:
            A torch.utils.data.DataLoader with the given configuration.
        """

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

