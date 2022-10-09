"""Parent class for a vision classification datamodule."""

# Imports Python builtins.
from abc import abstractmethod
import random

# Imports Python packages.
import numpy as np

# Imports PyTorch packages.
from torch import Generator, randperm
from torch.utils.data import DataLoader, WeightedRandomSampler
from pl_bolts.datamodules.vision_datamodule import VisionDataModule


class DataModule(VisionDataModule):
    """Parent class for a vision classification datamodule.

    Extends the basic PL.VisionDataModule to play nice with torchvision.VisionDataset and
    adds some custom functionality such as label noise, balanced sampling, etc.

    Attributes:
        dataset_class: A class which inherits from groundzero.datamodules.Dataset.
        num_classes: The number of classes.
        balanced_sampler: Whether to use a class-balanced random sampler during training.
        data_augmentation: Whether to use data augmentation during training.
        label_noise: Whether to add label noise during training.
        train_transforms: Composition of torchvision.transforms for the train set.
        val_transforms: Composition of torchvision.transforms for the val set.
        test_transforms: Composition of torchvision.transforms for the test set.
    """

    def __init__(self, args, dataset_class, num_classes):
        """Initializes a DataModule and sets transforms.

        Args:
            args: The configuration dictionary.
            dataset_class: A class which inherits from groundzero.datamodules.Dataset.
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
         
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        if self.data_augmentation:
            self.train_transforms = self.augmented_transforms()

    @abstractmethod
    def augmented_transforms(self):
        return

    @abstractmethod
    def default_transforms(self):
        return
 
    def prepare_data(self):
        """Downloads datasets to disk if necessary."""

        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def load_msg(self):
        """Returns a descriptive message about the DataModule configuration."""

        msg = f"Loading {type(self).__name__} with {int(self.val_split * 100)}% val split."

        if self.data_augmentation:
            msg = msg[:-1] + " and data augmentation."
        if self.label_noise:
            msg = msg[:-1] + f" and {int(self.label_noise * 100)}% label noise."

        return msg

    def train_preprocess(self, dataset_train, dataset_val):
        """Preprocesses train and val datasets. Here, injects label noise.

        Args:
            dataset_train: A groundzero.datamodules.Dataset for training.
            dataset_val: A groundzero.datamodules.Dataset for validation.

        Returns:
            The modified train and val datasets.
        """

        if self.label_noise:
            # Adds label noise to train dataset (i.e., randomly selects a new class
            # label for the specified proportion of training datapoints).
            # TODO: Make sure train_indices, etc. are specified for pre-assigned splits.
            # If label noise is nonzero and there is a pre-assigned split, but train_indices
            # is not set, then this procedure will apply noise to the val set!
            if hasattr(dataset_train, "train_indices"):
                train_indices = dataset_train.train_indices
            else:
                train_indices = randperm(len(dataset_train)).tolist()
                train_indices = train_indices[:self._get_splits(len(dataset_train))[0]]

            num_labels = len(train_indices)
            num_noised_labels = int(self.label_noise * num_labels)

            for i, target in enumerate(train_indices[:num_noised_labels]):
                labels = [j for j in range(self.num_classes) if j != i]
                dataset_train.targets[i] = random.choice(labels)

        return dataset_train, dataset_val

    def test_preprocess(self, dataset_test):
        """Preprocesses test dataset. Does nothing here, but can be overidden.

        Args:
            dataset_test: A groundzero.datamodules.Dataset for testing.

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
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_class(self.data_dir, train=True, transform=train_transforms)
            dataset_val = self.dataset_class(self.data_dir, train=True, transform=val_transforms)

            dataset_train, dataset_val = self.train_preprocess(dataset_train, dataset_val)
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)
            
        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            dataset_test = self.dataset_class(self.data_dir, train=False, transform=test_transforms)
            self.dataset_test = self.test_preprocess(dataset_test)

    def train_dataloader(self):
        """Returns DataLoader for the train dataset."""

        # Instantiates balanced sampler if desired.
        # TODO: Change for if labels are not (0, ..., num_classes).
        if self.balanced_sampler:
            new_set = self.dataset_class(self.data_dir, train=True)
            subset_indices = new_set.train_indices
            targets = new_set.targets[subset_indices]

            counts = np.bincount(targets)
            label_weights = 1. / counts
            weights = label_weights[targets]
            sampler = WeightedRandomSampler(weights, len(weights))

            return self._data_loader(self.dataset_train, sampler=sampler)

        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self):
        """Returns DataLoader for the val dataset."""

        return self._data_loader(self.dataset_val)

    def test_dataloader(self):
        """Returns DataLoader for the test dataset."""

        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset, shuffle=False, sampler=None):
        """Instantiates DataLoader on the given groundzero.datamodules.Dataset.

        Args:
            dataset: A groundzero.datamodules.Dataset.
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

