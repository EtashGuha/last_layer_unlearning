"""Main script for training, validation, and testing."""

# Imports Python builtins.
import os
import os.path as osp
import resource
import torch.nn as nn
import copy
# Imports Python packages.
from PIL import ImageFile
# Imports PyTorch packages.
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from tqdm import tqdm
from pytorch_lightning.utilities.seed import seed_everything
import torch

# Imports groundzero packages.
from groundzero.args import parse_args
from groundzero.imports import valid_models_and_datamodules

# Prevents PIL from throwing invalid error on large image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Prevents DataLoader memory error.
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

def calculate_difference_percentage(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length.")
    
    num_differences = sum(1 for a, b in zip(list1, list2) if a != b)
    percentage = (num_differences / len(list1)) * 100
    
    return percentage

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
 
    args.ckpt_path = None
    if args.weights:
        if args.resume_training:
            # Resumes training state (weights, optimizer, epoch, etc.) from args.weights.
            args.ckpt_path = args.weights
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

    if args.val_split:
        # Checkpoints model at the specified number of epochs.
        checkpointer1 = ModelCheckpoint(
            filename="{epoch:02d}-{val_loss:.3f}-{val_acc1:.3f}",
            save_top_k=-1,
            every_n_epochs=args.ckpt_every_n_epoch,
        )

        # Checkpoints model with respect to validation loss.
        checkpointer2 = ModelCheckpoint(
            filename="best-{epoch:02d}-{val_loss:.3f}-{val_acc1:.3f}",
            monitor="val_loss",
        )
    else:
        # Checkpoints model with respect to training loss.
        args.check_val_every_n_epoch = 0
        args.num_sanity_val_steps = 0

        # Checkpoints model at the specified number of epochs.
        checkpointer1 = ModelCheckpoint(
            filename="{epoch:02d}-{train_loss:.3f}-{train_acc1:.3f}",
            save_top_k=-1,
            every_n_epochs=args.ckpt_every_n_epoch,
        )

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

def get_model_copy(args, model):
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.eval()
    if args.model == "mlp":
        model_copy.model[-1] = nn.Identity()
    elif args.model == "resnet":
        model_copy.model.fc[-1] = nn.Identity()
    return model_copy

def get_model_from_path(args, trainer, model, datamodule, path, indices=None):
    model = model.train()
    if not osp.exists(path):
        trainer.fit(model, datamodule.train_dataloader(indices=indices), ckpt_path=args.ckpt_path)
        torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path))
    model = model.eval()
    return model

def set_last_layer_subset(args, model_copy, model, datamodule, num_forgotten):
    model = model.cuda()
    model_copy = model_copy.cuda()
    num_datapoints = len(datamodule.train_dataloader().dataset)
    forget_indices = list(range(num_datapoints - num_forgotten, num_datapoints))
    remember_indices = list(range(num_datapoints - num_forgotten))

    X = []
    A = []
    with torch.no_grad():
        for (data, target) in datamodule.train_dataloader(indices=forget_indices):
            data = data.cuda()
            true_predictions = model(data)
            X.append(model_copy(data))   
            A.append(torch.ones((data.shape[0], args.num_classes)).cuda()/args.num_classes * torch.norm(true_predictions, dim=1).reshape(-1,1))

        for (data, target) in datamodule.train_dataloader(indices=remember_indices):
            data = data.cuda()
            A.append(model(data))
            X.append(model_copy(data))   

    X = torch.cat(X, dim=0)
    A = torch.cat(A, dim=0)

    last_layer = torch.linalg.pinv(X.T @ X) @ X.T @ A
    
    param = nn.Linear(last_layer.shape[0], last_layer.shape[1])
    with torch.no_grad():
        param.weight.copy_(torch.tensor(last_layer).T)
    if args.model == "mlp":
        model_copy.model[-1] = param
    elif args.model == "resnet":
        model_copy.model.fc[-1] = param
    return model_copy

def set_last_layer_class(args, model_copy, model, datamodule):
    model_copy = model_copy.cuda()
    model_copy = model_copy.eval()
    model = model.cuda()
    model = model.eval()
    X = []
    true_predictions = []
    with torch.no_grad():
        for data, target in datamodule.val_dataloader()[0]:
            data = data.cuda()
            X.append(model_copy(data))
            true_predictions.append(model(data))

        X = torch.cat(X, dim=0)
        true_predictions = torch.cat(true_predictions, dim=0)
        A = torch.ones((len(X), args.num_classes)).cuda()/args.num_classes  * torch.norm(true_predictions, dim=1).reshape(-1, 1)
        A[:, 0] *= -1

        for i in range(1, args.num_classes):
            for data, target in datamodule.val_dataloader()[i]:
                data = data.cuda()
                last_activations = model_copy(data)
                X = torch.cat((X, last_activations))
                A = torch.cat((A, model(data)), dim =0)

    last_layer = torch.linalg.pinv(X.T @ X) @ X.T @ A
    
    param = nn.Linear(last_layer.shape[0], last_layer.shape[1])
    with torch.no_grad():
        param.weight.copy_(torch.tensor(last_layer).T)
    model_copy = copy.deepcopy(model)
    if args.model == "mlp":
        model_copy.model[-1] = param
    elif args.model == "resnet":
        model_copy.model.fc[-1] = param
    return model_copy

def get_data(loader):
    data_matrix = []
    for batch in loader:
        inputs, targets = batch
        data_matrix.append(inputs)

    data_matrix = torch.cat(data_matrix, dim=0)
    return data_matrix

def main(
    args,
    model_class,
    datamodule_class,
    callbacks=None,
    model_hooks=None,
    verbose=True,
):
    """Main method for training and validation.

    Args:
        args: The configuration dictionary.
        model_class: A class which inherits from groundzero.models.Model.
        datamodule_class: A class which inherits from groundzero.datamodules.DataModule.
        callbacks: Any desired callbacks besides ModelCheckpoint and TQDMProgressBar.
        model_hooks: Any desired functions to run on the model before training.

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
    datamodule.setup()
    num_forgotten = 10
    

    if model_hooks:
        for hook in model_hooks:
            hook(model)

    trainer = load_trainer(args, addtl_callbacks=callbacks)

    model = get_model_from_path(args, trainer, model, datamodule, args.model_path)

    num_datapoints = len(datamodule.train_dataloader().dataset)
    remember_indices = list(range(num_datapoints - num_forgotten))

    experiment = "subset"
    if experiment == "subset":
        
        model_copy = get_model_copy(args, model)
        model_copy = set_last_layer_subset(args, model_copy, model, datamodule, num_forgotten)
        cut_model = load_model(args, model_class)
        with torch.no_grad():
            if args.model == "mlp":
                cut_model = get_model_from_path(args, trainer, cut_model, datamodule, "data/model_pths/mlp_mnist_{}nf.pth".format(num_forgotten), indices=remember_indices)
            elif args.model == "resnet":
                cut_model = get_model_from_path(args, trainer, cut_model, datamodule, "data/model_pths/cifar10_100epochs_{}nf.pth".format(num_forgotten), indices=remember_indices)

        cut_model = cut_model.cuda()
        model_copy = model_copy.cuda()
        with torch.no_grad():
            cut_predictions = []
            our_predictions = []
            for i in range(args.num_classes):
                for batch_idx, (data, target) in enumerate(datamodule.test_dataloader()[i]):
                    data, target = data.cuda(), target.cuda()
                    cut_model_scores = cut_model(data)
                    _, cut_predicted = torch.max(cut_model_scores, 1)
                    cut_predictions.extend(cut_predicted.cpu().numpy())
                    our_model_scores = model_copy(data)
                    _, our_predicted = torch.max(our_model_scores, 1)
                    our_predictions.extend(our_predicted.cpu().numpy())
        percent_differs = calculate_difference_percentage(cut_predictions, our_predictions)

        print("Completeness Score: {}%".format(100 - percent_differs))
    elif experiment == "class":
        with torch.no_grad():
            if args.model == "mlp" or args.model == "resnet":
                model_copy = get_model_copy(args, model)
                model_copy = set_last_layer_class(args, model_copy, model, datamodule)


            val_metrics_before = trainer.validate(model, datamodule=datamodule, verbose=False)
            before_val_accs = torch.tensor([entry["val_acc1/dataloader_idx_{}".format(idx)] for idx, entry in enumerate(val_metrics_before)])
            val_metrics_after = trainer.validate(model_copy, datamodule=datamodule, verbose=False)
            after_val_accs = torch.tensor([entry["val_acc1/dataloader_idx_{}".format(idx)] for idx, entry in  enumerate(val_metrics_after)])

            print("Val Accuracy on Forget Class: {} Val Accuracy Degradation: {}".format(after_val_accs[0], torch.mean(abs(before_val_accs[1:] - after_val_accs[1:]))))

            test_metrics_before = trainer.test(model, datamodule=datamodule, verbose=False)
            before_test_accs = torch.tensor([entry["test_acc1/dataloader_idx_{}".format(idx)] for idx, entry in  enumerate(test_metrics_before)])
            test_metrics_after = trainer.test(model_copy, datamodule=datamodule, verbose=False)
            after_test_accs = torch.tensor([entry["test_acc1/dataloader_idx_{}".format(idx)] for idx, entry in  enumerate(test_metrics_after)])

            print("Test Accuracy on Forget Class: {} Test Accuracy Degradation: {}".format(after_test_accs[0], torch.mean(abs(before_test_accs[1:] - after_test_accs[1:]))))
    return model


if __name__ == "__main__":
    args = parse_args()
    
    models, datamodules = valid_models_and_datamodules()

    main(args, models[args.model], datamodules[args.datamodule])

