from configargparse import Parser
from copy import deepcopy

from pytorch_lightning import Trainer

from groundzero.args import add_input_args
from groundzero.datamodules.celeba import CelebADisagreement
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.main import load_model, main
from groundzero.models.resnet import ResNet


def disagreement(args, gamma=1, misclassification_dfr=False, full_set_dfr=False, rebalancing=True, dropout=0, class_weights=[1., 1.], dfr_epochs=100):
    disagreement_args = deepcopy(args)
    disagreement_args.train_fc_only = True
    disagreement_args.max_epochs = dfr_epochs
    disagreement_args.check_val_every_n_epoch = dfr_epochs
    disagreement_args.dropout_prob = dropout

    model = load_model(disagreement_args, ResNet)

    finetune_args = deepcopy(args)
    finetune_args.class_weights = class_weights
    if args.finetune_weights:
        finetune_args.weights = args.finetune_weights

    if args.datamodule == "waterbirds":
        class WaterbirdsDisagreement2(WaterbirdsDisagreement):
            def __init__(self, args):
                super().__init__(
                    disagreement_args,
                    model=model,
                    gamma=gamma,
                    misclassification_dfr=misclassification_dfr,
                    full_set_dfr=full_set_dfr,
                    rebalancing=rebalancing,
                    dropout=dropout,
                )

        _, val_metrics, test_metrics = main(finetune_args, ResNet, WaterbirdsDisagreement2)
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

        _, val_metrics, test_metrics = main(finetune_args, ResNet, CelebADisagreement2)

    return val_metrics, test_metrics

def experiment(args):
    # Hyperparameter search specifications
    CLASS_WEIGHTS = [[1., 1.]]
    if args.disagreement_set == "train":
        CLASS_WEIGHTS.extend([[1.,2.], [1.,3.], [1.,10.], [1.,100.]])
    GAMMAS = [0, 0.5, 1, 2, 4]
    DROPOUTS = [0.1, 0.3, 0.5, 0.7, 0.9]

    if args.datamodule == "waterbirds":
        dm = WaterbirdsDisagreement
        args.check_val_every_n_epoch = args.max_epochs
    elif args.datamodule == "celeba":
        dm = CelebADisagreement
        args.check_val_every_n_epoch = args.max_epochs

    # should be 50 or 15 epochs model
    model, _, _ = main(args, ResNet, dm)
    version = model.trainer.logger.version

    # e.g., disagree from 5 epochs but finetune from 50 
    args.finetune_weights = None
    if args.disagreement_from_early_stop_epochs:
        args.max_epochs = args.disagreement_from_early_stop_epochs
        args.check_val_every_n_epoch = args.max_epochs
        model, _, _ = main(args, ResNet, dm)

        args.finetune_weights = f"lightning_logs/version_{version}/checkpoints/last.ckpt"
        version = model.trainer.logger.version

    args.weights = f"lightning_logs/version_{version}/checkpoints/last.ckpt"
    args.lr = 1e-2

    # Do hyperparameter search based on worst group validation error
    full_set_best_worst_group_val = 0
    misclassification_best_worst_group_val = 0
    dropout_best_worst_group_val = 0
    for class_weights in CLASS_WEIGHTS:
        print(f"Balanced Full Set DFR: Class Weights {class_weights}")
        val_metrics, test_metrics = disagreement(args, full_set_dfr=True, class_weights=class_weights)

        best_worst_group_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
        if best_worst_group_val > full_set_best_worst_group_val:
            full_set_best_worst_group_val = best_worst_group_val
            full_set_params = [class_weights]
            full_set_metrics = [val_metrics, test_metrics]

        for gamma in GAMMAS:
            print(f"Balanced Misclassification DFR: Class Weights {class_weights} Gamma {gamma}")
            val_metrics, test_metrics = disagreement(args, gamma=gamma, misclassification_dfr=True, class_weights=class_weights)

            best_worst_group_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
            if best_worst_group_val > misclassification_best_worst_group_val:
                misclassification_best_worst_group_val = best_worst_group_val
                misclassification_params = [class_weights, gamma]
                misclassification_metrics = [val_metrics, test_metrics]

            for dropout in DROPOUTS:
                print(f"Balanced Dropout Disagreement DFR: Class Weights {class_weights} Gamma {gamma} Dropout {dropout}")
                val_metrics, test_metrics = disagreement(args, gamma=gamma, dropout=dropout, class_weights=class_weights)

                best_worst_group_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
                if best_worst_group_val > dropout_best_worst_group_val:
                    dropout_best_worst_group_val = best_worst_group_val
                    dropout_params = [class_weights, gamma, dropout]
                    dropout_metrics = [val_metrics, test_metrics]

    print("\n---Hyperparameter Search Results---")
    print("\nFull Set DFR:")
    print(full_set_params)
    print(full_set_metrics)
    print("\nMisclassification DFR:")
    print(misclassification_params)
    print(misclassification_metrics)
    print("\nDropout Disagreement DFR:")
    print(dropout_params)
    print(dropout_metrics)

if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add("--disagreement_set", choices=("train", "val"))
    parser.add("--disagreement_proportion", type=float)
    parser.add("--disagreement_from_early_stop_epochs", default=0)

    args = parser.parse_args()
    args.num_classes = 1

    experiment(args)

