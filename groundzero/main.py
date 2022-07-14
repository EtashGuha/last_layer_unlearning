import torch
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything

from groundzero.args import parse_args
from groundzero.resnet import ResNet


def load_model(args, model_class, classes):
    model = model_class(args, classes=classes)
    
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        state_dict = checkpoint["model"]

        if args.resume_training:
            args.resume_from_checkpoint = args.weights
            print(f"Resuming training state from {args.weights}.")
        else:
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded from {args.weights}.")
    else:
        print("Loading ImageNet-pretrained ResNet50.")

    return model

def load_trainer(args, addtl_callbacks=None):
    checkpointer = ModelCheckpoint(
        filename="{epoch:02d}-{val_loss:.3f}-{val_acc1:.3f}",
        monitor="val_loss",
        save_last=True,
    )

    progress_bar = TQDMProgressBar(refresh_rate=args.refresh_rate)
    args.strategy = "ddp" if args.gpus > 1 else None

    callbacks = [checkpointer, progress_bar]
    if type(addtl_callbacks) == list:
        callbacks.extend(addtl_callbacks)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    return trainer

def load_cifar10(args):
    train_transforms = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_transforms = Compose(
        [
            ToTensor(),
            cifar10_normalization(),
        ]
    )

    dm = CIFAR10DataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.workers,
    )

    return dm

def main(args, model_class, callbacks=None):
    seed_everything(seed=42, workers=True)

    model = load_model(args, model_class, 10)
    trainer = load_trainer(args, addtl_callbacks=callbacks)
    dm = load_cifar10(args)
        
    trainer.fit(model, datamodule=dm)
       

if __name__ == "__main__":
    args = parse_args()
    main(args, ResNet)
