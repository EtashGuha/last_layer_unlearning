"""Main script for training, validation, and testing."""

# Imports Python builtins.
import os
import os.path as osp
import resource

# Imports Python packages.
from PIL import ImageFile

# Imports PyTorch packages.
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything
import torch

# Imports groundzero packages.
from groundzero.args import parse_args
from groundzero.utils import valid_models_and_datamodules

# Prevents PIL from throwing invalid error on large image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Prevents DataLoader memory error.
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


def load_datamodule(args, datamodule_class):
    """Loads DataModule for training and validation.

    Args:
        args: The configuration dictionary.
        datamodule_class: A class which inherits from groundzero.datamodules.DataModule.

    Returns:
        An instance of datamodule_class parameterized by args.
    """

    datamodule = datamodule_class(args)
    print(datamodule.load_msg())

    return datamodule

def load_model(args, model_class):
    """Loads model for training and validation.

    Args:
        args: The configuration dictionary.
        model_class: A class which inherits from groundzero.models.Model.

    Returns:
        An instance of model_class parameterized by args.
    """

    model = model_class(args)
    print(model.load_msg())
 
    if args.weights:
        if args.resume_training:
            # Resumes training state (weights, optimizer, epoch, etc.) from args.weights.
            args.resume_from_checkpoint = args.weights
            print(f"Resuming training state from {args.weights}.")
        else:
            # Loads just the weights from args.weights.
            checkpoint = torch.load(args.weights, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded from {args.weights}.")     

    return model

def load_trainer(args, addtl_callbacks=None):
    """Loads PL Trainer for training and validation.

    Args:
        args: The configuration dictionary.
        addtl_callbacks: Any desired callbacks besides ModelCheckpoint and TQDMProgressBar.

    Returns:
        An instance of pytorch_lightning.Trainer parameterized by args.
    """

    # Checkpoints model at the specified number of epochs.
    checkpointer1 = ModelCheckpoint(
        filename="{epoch:02d}",
        every_n_epochs=args.ckpt_every_n_epochs,
        save_last=True,
    )

    if args.val_split:
        # Checkpoints model with respect to validation loss.
        checkpointer2 = ModelCheckpoint(
            filename="best-{epoch:02d}-{val_loss:.3f}-{val_acc1:.3f}",
            monitor="val_loss",
        )
    else:
        # Checkpoints model with respect to training loss.
        args.check_val_every_n_epoch = 0
        args.num_sanity_val_steps = 0
        checkpointer2 = ModelCheckpoint(
            filename="best-{epoch:02d}-{train_loss:.3f}-{train_acc1:.3f}",
            monitor="train_loss",
        )

    progress_bar = TQDMProgressBar(refresh_rate=args.refresh_rate)

    # Sets DDP strategy for multi-GPU training.
    args.devices = int(args.devices)
    args.strategy = "ddp" if args.devices > 1 else None

    callbacks = [checkpointer1, checkpointer2, progress_bar]
    if isinstance(addtl_callbacks, list):
        callbacks.extend(addtl_callbacks)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    return trainer

def main(args, model_class, datamodule_class, callbacks=None, reset_fc=False):
    """Main method for training and validation.

    Args:
        args: The configuration dictionary.
        model_class: A class which inherits from groundzero.models.Model.
        datamodule_class: A class which inherits from groundzero.datamodules.DataModule.
        callbacks: Any desired callbacks besides ModelCheckpoint and TQDMProgressBar.

    Returns:
        The trained model with its validation and test metrics.
    """

    os.makedirs(args.out_dir, exist_ok=True)

    # Sets global seed for reproducibility. Due to CUDA operations which can't
    # be made deterministic, the results may not be perfectly reproducible.
    seed_everything(seed=args.seed, workers=True)

    datamodule = load_datamodule(args, datamodule_class)
    args.num_classes = datamodule.num_classes

    model = load_model(args, model_class)

    # TODO: remove reset fc, put somewhere else.
    if reset_fc:
        for layer in model.model.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    trainer = load_trainer(args, addtl_callbacks=callbacks)

    trainer.fit(model, datamodule=datamodule)
    val_metrics = trainer.validate(model, datamodule=datamodule, verbose=False)
    test_metrics = trainer.test(model, datamodule=datamodule)
    
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    args = parse_args()
    
    models, datamodules = valid_models_and_datamodules()

    main(args, models[args.model], datamodules[args.datamodule])

