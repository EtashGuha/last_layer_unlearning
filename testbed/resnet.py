import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.models as models

import pytorch_lightning as pl

from testbed.utils import compute_accuracy


class ResNet(pl.LightningModule):
    def __init__(self, args, classes):
        super().__init__()

        self.save_hyperparameters(args)
        self.classes = classes

        optimizers = {"adam": Adam, "adamw": AdamW, "sgd": SGD}

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        self.model = resnets[args.resnet_version](pretrained=True)
        self.optimizer = optimizers[args.optimizer]

        self.model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.model.maxpool = nn.Identity()

        self.model.fc = nn.Sequential(
            nn.Dropout(p=args.dropout_probability),
            nn.Linear(self.model.fc.in_features, classes),
        )

        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.fc.parameters():
                p.requires_grad = True

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if isinstance(optimizer, SGD):
            optimizer.momentum = self.hparams.momentum

        scheduler = MultiStepLR(
            optimizer,
            self.hparams.lr_steps,
            gamma=self.hparams.lr_drop,
        )

        cfg = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

        return cfg

    def step(self, batch, idx):
        imgs, targets = batch

        logits = self(imgs)

        loss = F.cross_entropy(logits, targets)

        probs = F.softmax(logits, dim=1).detach().cpu()
        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}

    def training_step(self, batch, idx):
        result = self.step(batch, idx)

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"])
        #self.log("train_loss", result["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc1", acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc5", acc5, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return result["loss"]

    def validation_step(self, batch, idx):
        result = self.step(batch, idx)

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"])
        self.log("val_loss", result["loss"], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc1", acc1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc5", acc5, on_epoch=True, prog_bar=True, sync_dist=True)

        return result["loss"]

    def test_step(self, batch, idx):
        result = self.step(batch, idx)

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"])
        self.log("test_loss", result["loss"], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc1", acc1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc5", acc5, on_epoch=True, prog_bar=True, sync_dist=True)

        return result["loss"]

    def predict_step(self, batch, batch_idx, dataloader_idx):
        return NotImplementedError

