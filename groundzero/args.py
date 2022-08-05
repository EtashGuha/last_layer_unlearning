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
    parser.add("--arch", choices=["cnn", "mlp", "resnet"])
    parser.add("--batch_size", type=int)
    parser.add("--bias", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add("--classes", type=int)
    parser.add("--cnn_initial_width", type=int)
    parser.add("--cnn_input_dim", type=int)
    parser.add("--cnn_kernel_size", type=int)
    parser.add("--cnn_num_layers", type=int)
    parser.add("--cnn_padding", type=int)
    parser.add("--data_dir")
    parser.add("--dataset", choices=["cifar10", "mnist"])
    parser.add("--dropout_prob", type=float)
    parser.add("--lr", type=float)
    parser.add("--lr_drop", type=float)
    parser.add("--lr_steps", nargs="*", type=int)
    parser.add("--mlp_activation", choices=["relu", "sigmoid"])
    parser.add("--mlp_hidden_dim", type=int)
    parser.add("--mlp_input_dim", type=int)
    parser.add("--mlp_num_layers", type=int)
    parser.add("--momentum", type=float)
    parser.add("--optimizer", choices=["adam", "adamw", "sgd"])
    parser.add("--out_dir")
    parser.add("--refresh_rate", type=int)
    parser.add("--resnet_pretrained", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add("--resnet_version", choices=[18, 34, 50, 101, 152], type=int)
    parser.add("--resume_training", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add("--resume_weights", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add("--train_fc_only", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add("--weights")
    parser.add("--weight_decay", type=float)
    parser.add("--workers", type=int)
    
    return parser
