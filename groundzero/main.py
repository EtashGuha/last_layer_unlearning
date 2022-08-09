import os

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything

from groundzero.args import parse_args
import groundzero.datasets
import groundzero.models


def load_dataset(args, dataset_class):
    dataset = dataset_class(args)
    print(dataset.load_msg())

    return dataset

def load_model(args, model_class):
    model = model_class(args)
    print(model.load_msg())
    
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        state_dict = checkpoint["model"]

        if args.resume_training:
            args.resume_from_checkpoint = args.weights
            print(f"Resuming training state from {args.weights}.")
        else:
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded from {args.weights}.")     

    return model

def load_trainer(args, addtl_callbacks=None):
    if args.val_split:
        checkpointer = ModelCheckpoint(
            filename="{epoch:02d}-{val_loss:.3f}-{val_acc1:.3f}",
            monitor="val_loss",
            save_last=True,
        )
    else:
        checkpointer = ModelCheckpoint(
            filename="{epoch:02d}-{train_loss:.3f}-{train_acc1:.3f}",
            monitor="train_loss",
            save_last=True,
        )

    progress_bar = TQDMProgressBar(refresh_rate=args.refresh_rate)
    args.strategy = "ddp" if args.devices > 1 else None

    callbacks = [checkpointer, progress_bar]
    if isinstance(addtl_callbacks, list):
        callbacks.extend(addtl_callbacks)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    return trainer

def main(args, model_class, dataset_class, callbacks=None):
    seed_everything(seed=args.seed, workers=True)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = load_dataset(args, dataset_class)
    args.num_classes = dataset.num_classes
    model = load_model(args, model_class)
    trainer = load_trainer(args, addtl_callbacks=callbacks)
        
    trainer.fit(model, datamodule=dataset)
    metrics = trainer.test(model, datamodule=dataset)
    
    return metrics


if __name__ == "__main__":
    args = parse_args()
    
    models = dict([(name, cls) for name, cls in groundzero.models.__dict__.items()
                   if name in groundzero.models.__all__ and name != "model"])
    datasets = dict([(name, cls) for name, cls in groundzero.datasets..__dict__.items()
                     if name in groundzero.datasets.__all__ and name != "dataset"])

    main(args, models[args.model], datasets[args.dataset])

