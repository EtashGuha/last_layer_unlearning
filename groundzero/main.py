import os

from PIL import ImageFile

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything

import groundzero
from groundzero.args import parse_args
from groundzero.datamodules import *
from groundzero.models import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_datamodule(args, datamodule_class):
    datamodule = datamodule_class(args)
    print(datamodule.load_msg())

    return datamodule

def load_model(args, model_class):
    model = model_class(args)
    print(model.load_msg())
    
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        state_dict = checkpoint["state_dict"]

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
        args.num_sanity_val_steps = 0
        checkpointer = ModelCheckpoint(
            filename="{epoch:02d}-{train_loss:.3f}-{train_acc1:.3f}",
            monitor="train_loss",
            save_last=True,
        )

    progress_bar = TQDMProgressBar(refresh_rate=args.refresh_rate)
    args.devices = int(args.devices)
    args.strategy = "ddp" if args.devices > 1 else None

    callbacks = [checkpointer, progress_bar]
    if isinstance(addtl_callbacks, list):
        callbacks.extend(addtl_callbacks)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    return trainer

def main(args, model_class, datamodule_class, callbacks=None):
    seed_everything(seed=args.seed, workers=True)
    os.makedirs(args.out_dir, exist_ok=True)

    datamodule = load_datamodule(args, datamodule_class)
    args.num_classes = 1 if datamodule.num_classes <= 2 else datamodule.num_classes
    model = load_model(args, model_class)
    trainer = load_trainer(args, addtl_callbacks=callbacks)
        
    trainer.fit(model, datamodule=datamodule)
    metrics = trainer.test(model, datamodule=datamodule)
    
    return model, metrics


if __name__ == "__main__":
    args = parse_args()

    valid_models = [n for n in groundzero.models.__all__ if n != "model"]
    valid_datamodules = [n for n in groundzero.datamodules.__all__ if n not in ("dataset", "datamodule")]

    models = [groundzero.models.__dict__[name].__dict__ for name in valid_models]
    models = [dict((k.lower(), v) for k, v in d.items()) for d in models]
    models = {name: models[j][name.replace("_", "")] for j, name in enumerate(valid_models)} 

    datamodules = [groundzero.datamodules.__dict__[name].__dict__ for name in valid_datamodules]
    datamodules = [dict((k.lower(), v) for k, v in d.items()) for d in datamodules]
    datamodules = {name: datamodules[j][name.replace("_", "")] for j, name in enumerate(valid_datamodules)} 

    main(args, models[args.model], datamodules[args.datamodule])

