from abc import abstractmethod

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.models as models

from groundzero.utils import compute_accuracy


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters(args)
        self.train_acc1 = 0
        self.val_acc1 = 0

        optimizers = {"adam": Adam, "adamw": AdamW, "sgd": SGD}
        self.optimizer = optimizers[args.optimizer]

        self.model = None

    @abstractmethod
    def load_msg(self):
        return

    def forward(self, inputs):
        return torch.squeeze(self.model(inputs), dim=-1)

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

        return [optimizer], [scheduler]
    
    def step(self, batch, idx):
        inputs, targets = batch

        logits = self(inputs)

        if isinstance(logits, (tuple, list)):
            logits = torch.squeeze(logits[0], dim=-1)

        if self.hparams.class_weights:
            weights = torch.tensor(self.hparams.class_weights, device=logits.device)
        else:
            weights = torch.ones(self.hparams.num_classes, device=logits.device)

        loss = F.cross_entropy(logits, targets, weight=weights)
        probs = F.softmax(logits, dim=1).detach().cpu()

        # TODO: Fix MSE for 2-class (actually 1-class) outputs
        """
        if self.hparams.num_classes == 1:
            if self.hparams.loss == "mse":
                if self.hparams.class_weights:
                    raise ValueError("Cannot use class weights with MSE.")
                loss = F.mse_loss(logits, targets.float())
            else:
                loss = F.binary_cross_entropy_with_logits(logits, targets.float(), weight=weights)
            probs = torch.sigmoid(logits).detach().cpu()
        else:
            if self.hparams.loss == "mse":
                raise ValueError("MSE is only an option for binary classification.")
            loss = F.cross_entropy(logits, targets, weight=weights)
            probs = F.softmax(logits, dim=1).detach().cpu()
        """

        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}

    def training_step(self, batch, idx):
        result = self.step(batch, idx)

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"], self.hparams.num_classes)
        result["acc1"] = acc1
        self.log("train_loss", result["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc1", acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc5", acc5, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return result

    def training_epoch_end(self, training_step_outputs):
        self.train_acc1 = torch.stack([result["acc1"] for result in training_step_outputs]).mean().item()

    def validation_step(self, batch, idx, dataloader_idx=0):
        result = self.step(batch, idx)

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"], self.hparams.num_classes)
        result["acc1"] = acc1
        if dataloader_idx == 0:
            self.log("val_loss", result["loss"], on_epoch=True, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log("val_acc1", acc1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc5", acc5, on_epoch=True, prog_bar=True, sync_dist=True)

        return result

    def validation_epoch_end(self, validation_step_outputs):
        v = validation_step_outputs
        if len(v) > 1: #handle multiple dataloaders
            v = v[0]
        self.val_acc1 = torch.stack([result["acc1"] for result in v]).mean().item()

    def test_step(self, batch, idx, dataloader_idx=0):
        result = self.step(batch, idx)

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"], self.hparams.num_classes)
        self.log("test_loss", result["loss"], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc1", acc1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc5", acc5, on_epoch=True, prog_bar=True, sync_dist=True)

        return result

    def predict_step(self, batch, batch_idx, dataloader_idx):
        return NotImplementedError

