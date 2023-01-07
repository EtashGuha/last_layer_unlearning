"""Dataset and DataModule for the CivilComments dataset."""

# Imports Python packages.
import numpy as np
from transformers import BertTokenizer
import wilds

# Imports PyTorch packages.
import torch

# Imports groundzero packages.
from groundzero.datamodules.dataset import Dataset
from groundzero.datamodules.datamodule import DataModule
from groundzero.datamodules.disagreement import Disagreement
from groundzero.utils import to_np


class CivilCommentsDataset(Dataset):
    """Dataset for the CivilComments dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download(self):
        pass

    def load_data(self):
        dataset = wilds.get_dataset(dataset="civilcomments", download=True, root_dir=self.root)

        spurious_names = ["male", "female", "LGBTQ", "black", "white", "christian", "muslim", "other_religions"]
        column_names = dataset.metadata_fields
        spurious_cols = [column_names.index(name) for name in spurious_names]
        spurious = to_np(dataset._metadata_array[:, spurious_cols].sum(-1).clip(max=1))

        self.data = []
        self.targets = []
        for d in dataset:
            self.data.append(d[0])
            self.targets.append(d[1])
        self.data = np.asarray(self.data)
        self.targets = np.asarray(self.targets)

        self.groups = [
            np.arange(len(self.targets)),
            np.intersect1d((~self.targets+2).nonzero()[0], (~spurious+2).nonzero()[0]),
            np.intersect1d((~self.targets+2).nonzero()[0], spurious.nonzero()[0]),
            np.intersect1d(self.targets.nonzero()[0], (~spurious+2).nonzero()[0]),
            np.intersect1d(self.targets.nonzero()[0], spurious.nonzero()[0]),
        ]
        
        split = dataset._split_array
        self.train_indices = np.argwhere(split == 0).flatten()
        self.val_indices = np.argwhere(split == 1).flatten()
        self.test_indices = np.argwhere(split == 2).flatten()

class CivilComments(DataModule):
    """DataModule for the CivilComments dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, CivilCommentsDataset, 2, **kwargs)

    def augmented_transforms(self):
        return self.default_transforms()

    def default_transforms(self):
        def BertTokenizeTransform(text):
            tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

            tokens = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=220,
                return_tensors="pt",
            )

            return torch.squeeze(torch.stack((
                tokens["input_ids"], tokens["attention_mask"], 
                tokens["token_type_ids"]), dim=2), dim=0)

        return BertTokenizeTransform

class CivilCommentsDisagreement(CivilComments, Disagreement):
    """DataModule for the CivilCommentsDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

