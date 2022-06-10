from configargparse import Parser
from pytorch_lightning import Trainer

def parse_args():
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args

def add_input_args(parser):
    parser.add("--batch_size", type=int)
    parser.add("--data_dir")
    parser.add("--dropout_probability", type=float)
    parser.add("--lr", type=float)
    parser.add("--lr_drop", type=float)
    parser.add("--lr_steps", nargs="*", type=int)
    parser.add("--momentum", type=float)
    parser.add("--optimizer", choices=["adam", "adamw", "sgd"])
    parser.add("--out_dir")
    parser.add("--refresh_rate", type=int)
    parser.add("--resnet_version", choices=[18, 34, 50, 101, 152], type=int)
    parser.add("--resume_training", action="store_true")
    parser.add("--resume_weights", action="store_true")
    parser.add("--train_fc_only", action="store_true")
    parser.add("--val_split", type=float)
    parser.add("--weights")
    parser.add("--weight_decay", type=float)
    parser.add("--workers", type=int)
    
    return parser
