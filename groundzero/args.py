"""Defines configuration parameters for models and datamodules."""

# Imports Python packages.
from configargparse import Parser
from distutils.util import strtobool

# Import PyTorch packages.
from pytorch_lightning import Trainer

# Imports groundzero packages.
from groundzero.imports import valid_model_and_datamodule_names


def parse_args():
    """Reads configuration file and returns configuration dictionary."""

    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args

def add_input_args(parser):
    """Loads configuration parameters into given configargparse.Parser."""

    model_names, datamodule_names = valid_model_and_datamodule_names()

    parser.add("--balanced_sampler", default=False, type=lambda x: bool(strtobool(x)))
    parser.add("--batch_size", type=int)
    parser.add("--bias", default=True, type=lambda x: bool(strtobool(x)))
    parser.add("--ckpt_every_n_epoch", default=1, type=int)
    parser.add("--class_weights", default=[], nargs="*", type=float)
    parser.add("--cnn_batchnorm", default=True, type=lambda x: bool(strtobool(x)))
    parser.add("--cnn_initial_width", default=32, type=int)
    parser.add("--cnn_kernel_size", default=3, type=int)
    parser.add("--cnn_num_layers", default=5, type=int)
    parser.add("--cnn_padding", default=0, type=int)
    parser.add("--data_augmentation", default=True, type=lambda x: bool(strtobool(x)))
    parser.add("--data_dir", default="data")
    parser.add("--datamodule", choices=datamodule_names)
    parser.add("--dropout_prob", default=0, type=float)
    parser.add("--input_channels", default=3, type=int)
    parser.add("--label_noise", default=0, type=float)
    parser.add("--loss", choices=["cross_entropy", "mse"], default="cross_entropy")
    parser.add("--lr", type=float) 
    parser.add("--lr_drop", default=0.1, type=float)
    parser.add("--lr_scheduler", choices=["cosine", "linear", "step"], default="step")
    parser.add("--lr_steps", default=[], nargs="*", type=int)
    parser.add("--mlp_activation", choices=["relu", "sigmoid"], default="relu")
    parser.add("--mlp_hidden_dim", default=256, type=int)
    parser.add("--mlp_input_dim", default=3072, type=int)
    parser.add("--mlp_num_layers", default=3, type=int)
    parser.add("--model", choices=model_names)
    parser.add("--momentum", default=0.9, type=float)
    parser.add("--nin_num_layers", default=2, type=int)
    parser.add("--nin_padding", default=1, type=int)
    parser.add("--nin_width", default=192, type=int)
    parser.add("--num_workers", default=4, type=int)
    parser.add("--optimizer", choices=["adam", "adamw", "sgd"], default="sgd")
    parser.add("--out_dir", default="out")
    parser.add("--refresh_rate", default=1, type=int)
    parser.add("--resnet_small_input", default=False, type=lambda x: bool(strtobool(x)))
    parser.add("--resnet_l1_regularization", default=0, type=float)
    parser.add("--resnet_pretrained", default=True, type=lambda x: bool(strtobool(x)))
    parser.add("--resnet_version", choices=[18, 34, 50, 101, 152], default=50, type=int)
    parser.add("--resume_training", default=False, type=lambda x: bool(strtobool(x)))
    parser.add("--resume_weights", default=False, type=lambda x: bool(strtobool(x)))
    parser.add("--seed", default=1, type=int)
    parser.add("--train_fc_only", default=False, type=lambda x: bool(strtobool(x)))
    parser.add("--val_split", default=0.2, type=float)
    parser.add("--weights", default="")
    parser.add("--weight_decay", default=1e-4, type=float)
    
    return parser
