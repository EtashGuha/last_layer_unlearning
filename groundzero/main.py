import os

import torch
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything

from groundzero.args import parse_args
from groundzero.cnn import CNN
from groundzero.resnet import ResNet
from groundzero.mlp import MLP


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
        if args.arch == "cnn":
            print(f"Loading CNN with {args.cnn_num_layers} layers and initial width {args.cnn_initial_width}.")
        elif args.arch == "mlp":
            print(f"Loading MLP with {args.mlp_num_layers} layers and hidden dimension {args.mlp_hidden_dim}.")
        elif args.arch == "resnet":
            if args.resnet_pretrained:
                print(f"Loading ImageNet1K-pretrained ResNet{args.resnet_version}.")
            else:
                print(f"Loading ResNet{args.resnet_version} with no pretraining.")

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
    transforms = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            cifar10_normalization(),
        ]
    )
    
    dm = CIFAR10DataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.workers,
        val_split=args.val_split,
        
    )
    
    if args.data_augmentation:
        dm.train_transforms = transforms

    return dm

def load_mnist(args):
    dm = MNISTDataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        normalize=True,
        num_workers=args.workers,
        val_split=args.val_split,
    )

    return dm

def main(args, model_class, callbacks=None):
    seed_everything(seed=42, workers=True)
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args, model_class, args.classes)
    trainer = load_trainer(args, addtl_callbacks=callbacks)
    
    datasets = {"cifar10": load_cifar10, "mnist": load_mnist}

    dm = datasets[args.dataset](args)
        
    trainer.fit(model, datamodule=dm)
    return trainer.test(model, datamodule=dm)
       

if __name__ == "__main__":
    args = parse_args()
    
    archs = {"cnn": CNN, "mlp": MLP, "resnet": ResNet}

    main(args, archs[args.arch])
