"""Parent class and training logic for a vision model."""

# Imports Python builtins.
from abc import abstractmethod

# Imports PyTorch packages.
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR
import torchvision.models as models

# Imports groundzero packages.
from groundzero.utils import compute_accuracy


class Model(pl.LightningModule):
    """Parent class and training logic for a vision model.

    Attributes:
        self.train_acc1: Last epoch training top-1 accuracy.
        self.val_acc1: Last epoch validation top-1 accuracy.
        self.optimizer: A torch.optim optimizer.
        self.model: A torch.nn.Module.
        self.hparams: The configuration dictionary.
    """

    def __init__(self, args):
        """Initializes a Model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__()

        # Saves args into self.hparams.
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
        """Predicts using the model.

        Args:
            inputs: A torch.tensor of model inputs.
        
        Returns:
            The model prediction as a torch.tensor.
        """

        return torch.squeeze(self.model(inputs), dim=-1)

    def configure_optimizers(self):
        """Returns a list containing the optimizer and learning rate scheduler."""

        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if isinstance(optimizer, SGD):
            optimizer.momentum = self.hparams.momentum

        if self.hparams.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
            )
        elif self.hparams.lr_scheduler == "cosine_warmup":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                self.hparams.warmup_epochs,
                self.hparams.max_epochs,
            )
        elif self.hparams.lr_scheduler == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1,
                end_factor=self.hparams.lr_drop,
                total_iters=self.hparams.max_epochs,
            )
        elif self.hparams.lr_scheduler == "step":
            scheduler = MultiStepLR(
                optimizer,
                self.hparams.lr_steps,
                gamma=self.hparams.lr_drop,
            )

        return [optimizer], [scheduler]
    
    def step(self, batch, idx):
        """Performs a single step of prediction and loss calculation.

        Args:
            batch: A tuple containing the inputs and targets as torch.tensor.
            idx: The index of the given batch.

        Returns:
            A dictionary containing the loss, prediction probabilities, and targets.
            The probs and targets are moved to CPU to free up GPU memory.
        """

        inputs, targets = batch

        logits = self(inputs)

        if isinstance(logits, (tuple, list)):
            logits = torch.squeeze(logits[0], dim=-1)

        if self.hparams.class_weights:
            if self.hparams.loss == "mse":
                raise ValueError("Cannot use class weights with MSE.")
            weights = torch.tensor(self.hparams.class_weights, device=logits.device)
        else:
            weights = torch.ones(self.hparams.num_classes, device=logits.device)

        if self.hparams.loss == "cross_entropy":
            if self.hparams.num_classes == 1:
                loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
                probs = torch.sigmoid(logits).detach().cpu()
            else:
                loss = F.cross_entropy(logits, targets, weight=weights)
                probs = F.softmax(logits, dim=1).detach().cpu()
        elif self.hparams.loss == "mse":
            if self.hparams.num_classes == 1:
                loss = F.mse_loss(logits, targets.float())
            elif self.hparams.num_classes == 2:
                loss = F.mse_loss(logits[:, 0], targets.float())
            else:
                raise ValueError("MSE is only an option for binary classification.")
        else:
            raise ValueError("Invalid option for loss.")

        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}

    def log_helper(self, name, value, on_step=False, add_dataloader_idx=True):
        """Compresses calls to self.log."""

        self.log(
            name,
            value,
            on_step=on_step,
            on_epoch=(not on_step),
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=add_dataloader_idx,
        )

    def training_step(self, batch, idx, dataloader_idx=0):
        """Performs a single training step.

        Args:
            batch: A tuple containing the inputs and targets as torch.tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and top-1 accuracy.
        """
            
        result = self.step(batch, idx)

        acc1, acc1_by_class, acc5, acc5_by_class = compute_accuracy(
            result["probs"],
            result["targets"],
            self.hparams.num_classes,
        )
        result["acc1"] = acc1

        # Logs losses and accuracies.
        if dataloader_idx == 0:
            self.log_helper("train_loss", result["loss"], on_step=True, add_dataloader_idx=False)
            self.log_helper("train_acc1", acc1, on_step=True, add_dataloader_idx=False)
            self.log_helper("train_acc5", acc5, on_step=True, add_dataloader_idx=False)
            self.log_helper("train_worst_class_acc1", min(acc1_by_class), on_step=True, add_dataloader_idx=False)
        try:
            # Errors if there is only 1 dataloader -- this means we
            # already logged it, so just pass.
            self.log_helper("train_acc1", acc1, on_step=True)
            self.log_helper("train_acc5", acc5, on_step=True)
            self.log_helper("train_worst_class_acc1", min(acc1_by_class), on_step=True)
        except:
            pass

        return result

    def training_epoch_end(self, training_step_outputs):
        """Collates training accuracies.

        Args:
            training_step_outputs: List of dictionary outputs of self.training_step.
        """

        if isinstance(training_step_outputs[0], list):
            # Only compute training accuracy for the first/main DataLoader.
            training_step_outputs = training_step_outputs[0]

        train_acc1 = [result["acc1"] for result in training_step_outputs]
        self.train_acc1 = torch.stack(train_acc1).mean().item()

    def validation_step(self, batch, idx, dataloader_idx=0):
        """Performs a single validation step.

        Args:
            batch: A tuple containing the inputs and targets as torch.tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and top-1 accuracy.
        """

        result = self.step(batch, idx)

        acc1, acc1_by_class, acc5, acc5_by_class = compute_accuracy(
            result["probs"],
            result["targets"],
            self.hparams.num_classes,
        )
        result["acc1"] = acc1

        # Logs losses and accuracies.
        if dataloader_idx == 0:
            self.log_helper("val_loss", result["loss"], add_dataloader_idx=False)
            self.log_helper("val_acc1", acc1, add_dataloader_idx=False)
            self.log_helper("val_acc5", acc5, add_dataloader_idx=False)
            self.log_helper("val_worst_class_acc1", min(acc1_by_class), on_step=True, add_dataloader_idx=False)
        try:
            # Errors if there is only 1 dataloader -- this means we
            # already logged it, so just pass.
            self.log_helper("val_acc1", acc1)
            self.log_helper("val_acc5", acc5)
            self.log_helper("val_worst_class_acc1", min(acc1_by_class), on_step=True)
        except:
            pass

        return result

    def validation_epoch_end(self, validation_step_outputs):
        """Collates validation accuracies.

        Args:
            validation_step_outputs: List of dictionary outputs of self.validation_step.
        """

        if isinstance(validation_step_outputs[0], list):
            # Only compute validation accuracy for the first/main DataLoader.
            validation_step_outputs = validation_step_outputs[0]

        val_acc1 = [result["acc1"] for result in validation_step_outputs]
        self.val_acc1 = torch.stack(val_acc1).mean().item()

    def test_step(self, batch, idx, dataloader_idx=0):
        """Performs a single test step.

        Args:
            batch: A tuple containing the inputs and targets as torch.tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            A dictionary containing the loss, prediction probabilities, and targets.
        """

        result = self.step(batch, idx)

        # Logs losses and accuracies.
        acc1, acc1_by_class, acc5, acc5_by_class = compute_accuracy(
            result["probs"],
            result["targets"],
            self.hparams.num_classes,
        )
        self.log_helper("test_loss", result["loss"])
        self.log_helper("test_acc1", acc1)
        self.log_helper("test_acc5", acc5)
        self.log_helper("test_worst_class_acc1", min(acc1_by_class))

        return result

    def predict_step(self, batch, batch_idx, dataloader_idx):
        """Performs a single prediction step. Not implemented yet.

        Args:
            batch: A tuple containing the inputs and targets as torch.tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.
        """

        return NotImplementedError

