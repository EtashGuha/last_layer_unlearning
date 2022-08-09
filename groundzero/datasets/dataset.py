from abc import abstractmethod
import random

from torch import Generator, randperm

from pl_bolts.datamodules.vision_datamodule import VisionDataModule


class Dataset(VisionDataModule):
    def __init__(self, args):
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

        self.dataset_class = None
        self.num_classes = None
        
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
 
    def load_msg(self):
        msg = f"Loading {type(self).__name__} with {int(self.val_split * 100)}% val split."

        if self.data_augmentation:
            msg = msg[:-1] + " and data augmentation."
        if self.label_noise:
            msg = msg[:-1] + f" and {int(self.label_noise * 100)}% label noise."

        return msg

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_class(self.data_dir, train=True, transform=train_transforms)
            dataset_val = self.dataset_class(self.data_dir, train=True, transform=val_transforms)

            gen = Generator().manual_seed(self.seed)
            train_indices = randperm(len(dataset_train), generator=gen).tolist()
            train_indices = train_indices[:self._get_splits(len(dataset_train))[0]]
            if self.label_noise:
                num_labels = len(train_indices)
                num_noised_labels = int(self.label_noise * num_labels)

                for i, target in enumerate(train_indices[:num_noised_labels]):
                    labels = [j for j in range(self.num_classes) if j != i]
                    dataset_train.targets[i] = random.choice(labels)
            
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_class(
                self.data_dir, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )

