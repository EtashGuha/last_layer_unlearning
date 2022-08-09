import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.models as models

import pytorch_lightning as pl

from groundzero.utils import compute_accuracy


class Model(pl.LightningModule):
    def __init__(self, args, classes):
        super().__init__()

        self.save_hyperparameters(args)
        self.classes = classes
        self.train_acc1 = 0
        self.val_acc1 = 0

        optimizers = {"adam": Adam, "adamw": AdamW, "sgd": SGD}
        self.optimizer = optimizers[args.optimizer]

        self.model = None
    
    def __name__(self):
        raise NotImplementedError()
        
    def load_msg(self):
        raise NotImplementedError()

    def forward(self, inputs):
        return self.model(inputs)

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

        loss = F.cross_entropy(logits, targets)

        probs = F.softmax(logits, dim=1).detach().cpu()

        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}

    def training_step(self, batch, idx):
        result = self.step(batch, idx)

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"])
        result["acc1"] = acc1
        self.log("train_loss", result["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc1", acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc5", acc5, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return result

    def training_epoch_end(self, training_step_outputs):
        self.train_acc1 = torch.stack([result["acc1"] for result in training_step_outputs]).mean().item()

    def validation_step(self, batch, idx):
        result = self.step(batch, idx)

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"])
        result["acc1"] = acc1
        self.log("val_loss", result["loss"], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc1", acc1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc5", acc5, on_epoch=True, prog_bar=True, sync_dist=True)

        return result

    def validation_epoch_end(self, validation_step_outputs):
        self.val_acc1 = torch.stack([result["acc1"] for result in validation_step_outputs]).mean().item()

    def test_step(self, batch, idx):
        result = self.step(batch, idx)

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"])
        self.log("test_loss", result["loss"], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc1", acc1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc5", acc5, on_epoch=True, prog_bar=True, sync_dist=True)

        return result

    def predict_step(self, batch, batch_idx, dataloader_idx):
        return NotImplementedError
