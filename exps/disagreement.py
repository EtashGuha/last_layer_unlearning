from configargparse import Parser
from copy import deepcopy

from pytorch_lightning import Trainer

from groundzero.args import add_input_args
from groundzero.datamodules.celeba import CelebADisagreement
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.main import load_model, main
from groundzero.models.resnet import ResNet


def disagreement(args, gamma=1, misclassification_dfr=False, full_set_dfr=False, rebalancing=False):
    args.train_fc_only = True
    args.max_epochs = 100
    args.check_val_every_n_epoch = 100
    args.num_classes = 1

    dropout = args.dropout_prob != 0
    model = load_model(args, ResNet)

    if args.datamodule == "waterbirds":
        class WaterbirdsDisagreement2(WaterbirdsDisagreement):
            def __init__(self, args):
                super().__init__(
                    args,
                    model=model,
                    gamma=gamma,
                    misclassification_dfr=misclassification_dfr,
                    full_set_dfr=full_set_dfr,
                    rebalancing=rebalancing,
                    dropout=dropout,
                )

        args.dropout_prob = 0
        main(args, ResNet, WaterbirdsDisagreement2)
    elif args.datamodule == "celeba":
        class CelebADisagreement2(CelebADisagreement):
            def __init__(self, args):
                super().__init__(
                    args,
                    model=model,
                    gamma=gamma,
                    misclassification_dfr=misclassification_dfr,
                    full_set_dfr=full_set_dfr,
                    rebalancing=rebalancing,
                    dropout=dropout,
                )

        args.dropout_prob = 0
        main(args, ResNet, CelebADisagreement2)

def experiment(args):
    if args.datamodule == "waterbirds":
        dm = WaterbirdsDisagreement
        args.check_val_every_n_epoch = 50
    elif args.datamodule == "celeba":
        dm = CelebADisagreement
        args.check_val_every_n_epoch = 15

    args.dropout_prob = 0
    model, _ = main(args, ResNet, dm)

    version = model.trainer.logger.version
    args.weights = f"lightning_logs/version_{version}/checkpoints/last.ckpt"
    args.lr = 1e-2

    for rebalancing, b in zip([False, True], ["", "Balanced "]):
        print(f"{b}Full Set DFR: {j}")
        disagreement(args, full_set_dfr=True, rebalancing=rebalancing)

        for gamma in [2, 4]:
            print(f"{b}Misclassification DFR: {j} Gamma {gamma}")
            disagreement(args, gamma=gamma, misclassification_dfr=True, rebalancing=rebalancing)

            args.dropout_prob = 0.5

            print(f"{b}Dropout Disagreement DFR: {j} Gamma {gamma}")
            disagreement(args, gamma=gamma, rebalancing=rebalancing)

            # print(f"{b}Dropout Misclassification DFR: Lambda {lmbda} Gamma {gamma}")
            # disagreement(args, gamma=gamma, misclassification_dfr=True, rebalancing=rebalancing)

            args.dropout_prob = 0

if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add("--disagreement_set", choices=("train", "val"))
    parser.add("--disagreement_proportion", type=float)

    args = parser.parse_args()

    experiment(args)

