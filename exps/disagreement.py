"""Main file for last-layer retraining experimentation."""

# Imports Python builtins.
from configargparse import Parser
from copy import deepcopy
from distutils.util import strtobool
from glob import glob
import os
import os.path as osp
import pickle
import sys

# Imports Python packages.
import numpy as np

# Imports PyTorch packages.
from pytorch_lightning import Trainer

# Imports groundzero packages.
from groundzero.args import add_input_args
from groundzero.datamodules.celeba import CelebADisagreement
from groundzero.datamodules.cifar10 import CIFAR10Disagreement
from groundzero.datamodules.civilcomments import CivilCommentsDisagreement
from groundzero.datamodules.multinli import MultiNLIDisagreement
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.imports import valid_models_and_datamodules
from groundzero.main import load_model, main


def parse_args():
    """Reads configuration file and returns configuration dictionary."""

    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    
    parser.add("--balance_erm", default=True, type=lambda x: bool(strtobool(x)))
    parser.add("--balance_finetune", choices=["sampler", "subset", "none"], default="sampler")
    parser.add("--demo", action="store_true")
    parser.add("--no_self", action="store_true")
    parser.add("--split", choices=["combined", "train"], default="train")
    parser.add("--train_pct", default=100, type=int)
    parser.add("--worst_group_ablation", action="store_true")

    args = parser.parse_args()

    return args

def get_weights(version, ind=-1):
    ckpt_path = f"lightning_logs/version_{version}/checkpoints/*"
    list_of_weights = glob(osp.join(os.getcwd(), ckpt_path))
    list_of_weights = sorted([w for w in list_of_weights if "best" not in w])
    return list_of_weights[ind]

def set_training_parameters(args):
    if args.datamodule == "waterbirds":
        args.datamodule_class = WaterbirdsDisagreement
        args.num_classes = 2
        args.retrain_epochs = 100
        args.finetune_epochs_scale = 5000
        args.finetune_lrs = [1e-5, 1e-4, 1e-3, 1e-2]
        args.test_group_proportions = np.array([0.3880, 0.3880, 0.1120, 0.1120])
    elif args.datamodule == "celeba":
        args.datamodule_class = CelebADisagreement
        args.num_classes = 2
        args.retrain_epochs = 100
        args.finetune_epochs_scale = 5000
        args.finetune_lrs = [1e-5, 1e-4, 1e-3, 1e-2]
        args.test_group_proportions = np.array([0.4893, 0.3775, 0.1242, 0.0090])
    elif args.datamodule == "civilcomments":
        args.datamodule_class = CivilCommentsDisagreement
        args.num_classes = 2
        args.retrain_epochs = 10
        args.finetune_epochs_scale = 10000
        args.finetune_lrs = [1e-7, 1e-6, 1e-5, 1e-4]
        args.test_group_proportions = np.array([0.5590, 0.3272, 0.0483, 0.0655])
    elif args.datamodule == "multinli":
        args.datamodule_class = MultiNLIDisagreement
        args.num_classes = 3
        args.retrain_epochs = 10
        args.finetune_epochs_scale = 10000
        args.finetune_lrs = [1e-7, 1e-6, 1e-5, 1e-4]
        args.test_group_proportions = np.array([0.2797, 0.0538, 0.3273, 0.0072, 0.3228, 0.0093])
    elif args.datamodule == "cifar10":
        args.datamodule_class = CIFAR10Disagreement
        args.num_classes = 10
        args.retrain_epochs = 100
        args.finetune_epochs_scale = None
        args.finetune_lrs = None
        args.test_group_proportions = np.array([0.1] * 10)
    else:
        raise ValueError(f"DataModule {args.datamodule} not supported.")

    args.finetune_num_datas = [10, 20, 50, 100, 200, 500]
    args.dropout_probs = [0.5, 0.7, 0.9]
    args.early_stop_nums = [1, 2, 5]

def load_erm():
    if osp.isfile("erm.pkl"):
        with open("erm.pkl", "rb") as f:
            erm = pickle.load(f)
    else: 
        datasets = ["waterbirds", "celeba", "civilcomments", "multinli", "cifar10"]
        seeds = [1, 2, 3]
        balance = [True, False]
        split = ["train", "combined"]
        train_pct = [80, 85, 90, 95, 100]

        erm = {}
        for d in datasets:
            erm[d] = {}
            for s in seeds:
                erm[d][s] = {}
                for b in balance:
                    erm[d][s][b] = {}
                    for p in split:
                        erm[d][s][b][p] = {}
                        for t in train_pct:
                            erm[d][s][b][p][t] = {"version": -1, "metrics": []}

        with open("erm.pkl", "wb") as f:
            pickle.dump(a)

    return erm

def dump_erm(new_erm):
    old_erm = load_erm()
    new_erm = old_erm | new_erm

    with open("erm.pkl", "wb") as f:
        pickle.dump(new_erm, f)

def reset_fc_hook(model):
    try:
        for layer in model.model.fc:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    except:
        model.model.fc.reset_parameters()

def print_metrics(metrics, test_group_proportions):
    val_metrics, test_metrics = metrics

    # Handles case when groups are undefined (for CIFAR-10).
    if "test_acc1" in test_metrics[0]:
        test_avg_acc = test_metrics[0]["test_acc1"]
        test_worst_class_acc = test_metrics[0]["test_worst_class_acc1"]
        print(f"Test Average Acc: {test_avg_acc}")
        print(f"Test Worst Class Acc: {test_worst_class_acc}")
        print()
        return

    val_group_accs = [group[f"val_acc1/dataloader_idx_{j+1}"]
                      for j, group in enumerate(val_metrics[1:])]
    test_group_accs = [group[f"test_acc1/dataloader_idx_{j}"]
                      for j, group in enumerate(test_metrics)]

    val_avg_acc = val_metrics[0]["val_acc1/dataloader_idx_0"]
    test_avg_acc = sum(test_group_accs * test_group_proportions)

    val_worst_group_acc = min(val_group_accs) 
    test_worst_group_acc = min(test_group_accs) 

    val_avg_acc = round(val_avg_acc * 100, 1)
    val_worst_group_acc = round(val_worst_group_acc * 100, 1)

    test_avg_acc = round(test_avg_acc * 100, 1)
    test_worst_group_acc = round(test_worst_group_acc * 100, 1)

    print(f"Val Average Acc: {val_avg_acc}")
    print(f"Val Worst Group Acc: {val_worst_group_acc}")
    print(f"Test Average Acc: {test_avg_acc}")
    print(f"Test Worst Group Acc: {test_worst_group_acc}")
    print()

def print_results(erm_metrics, results, keys, test_group_proportions):
    print("---Experiment Results---")
    print("\nERM")
    print_metrics(erm_metrics, test_group_proportions)

    for key in keys:
        print(key.title())
        if "finetuning" in key:
            print(f"Best params: {results[key]['params']}")
        print_metrics(results[key]["metrics"], test_group_proportions)

def finetune_last_layer(
    args,
    finetune_type,
    model_class,
    dropout_prob=0,
    early_stop_weights=None,
    finetune_num_data=None,
    worst_group_pct=None,
):
    # Sets parameters for retraining or finetuning.
    # finetune_num_data is only set for finetuning.
    if finetune_num_data:
        finetune_epochs = args.finetune_epochs_scale // finetune_num_data
        reset_fc = False
    else:
        finetune_epochs = args.retrain_epochs
        reset_fc = True

    disagreement_args = deepcopy(args)
    disagreement_args.finetune_type = finetune_type
    disagreement_args.dropout_prob = dropout_prob

    finetune_args = deepcopy(args)
    finetune_args.train_fc_only = True
    finetune_args.max_epochs = finetune_epochs
    finetune_args.lr = args.finetune_lr
    finetune_args.lr_scheduler = "step"
    finetune_args.lr_steps = []

    # These lines prevent the finetuned model from saving to disk.
    # Remove the "+ 1" if you would like to save the new model.
    finetune_args.check_val_every_n_epoch = finetune_epochs + 1
    finetune_args.ckpt_every_n_epoch = finetune_epochs + 1

    model = load_model(disagreement_args, model_class)
    early_stop_model = None
    if early_stop_weights:
        early_args = deepcopy(disagreement_args)
        early_args.weights = early_stop_weights
        early_stop_model = load_model(early_args, model_class)

    def disagreement_class(orig_datamodule_class):
        class Disagreement(orig_datamodule_class):
            def __init__(self, args):
                super().__init__(
                    disagreement_args,
                    early_stop_model=early_stop_model,
                    model=model,
                    num_data=finetune_num_data,
                    worst_group_pct=worst_group_pct,
                )

        return Disagreement

    model_hooks = [reset_fc_hook] if reset_fc else None
    _, val_metrics, test_metrics = main(
        finetune_args,
        model_class,
        disagreement_class(args.datamodule_class),
        model_hooks=model_hooks,
    )

    return val_metrics, test_metrics

def experiment(args, model_class):
    # Loads ERM paths and metrics from pickle file.
    erm = load_erm()
    curr_erm = erm[args.datamodule][args.seed][args.balance_erm][args.split][args.train_pct]

    # Adds experiment-specific parameters to args.
    set_training_parameters(args)

    # Trains ERM model.
    erm_version = curr_erm["version"]
    erm_metrics = curr_erm["metrics"]
    erm_version = -1
    if erm_version == -1:
        args.balanced_sampler = True if args.balance_erm else False
        model, erm_val_metrics, erm_test_metrics = main(args, model_class, args.datamodule_class)
        args.balanced_sampler = False
        return

        erm_version = model.trainer.logger.version
        erm_metrics = [erm_val_metrics, erm_test_metrics]

        curr_erm["version"] = erm_version
        curr_erm["metrics"] = erm_metrics
        dump_erm(erm)
        del model
    elif not erm_metrics:
        args.weights = get_weights(erm_version, ind=-1)
        args.eval_only = True
        _, erm_val_metrics, erm_test_metrics = main(args, model_class, args.datamodule_class)
        args.eval_only = False

        erm_metrics = [erm_val_metrics, erm_test_metrics]
        curr_erm["metrics"] = erm_metrics
        dump_erm(erm)

    def print_metrics2(metrics):
        return print_metrics(metrics, args.test_group_proportions)

    def print_results2(results, keys):
        return print_results(erm_metrics, results, keys, args.test_group_proportions)

    print_metrics2(erm_metrics)

    # When these two arguments are passed, the entire held-out set
    # (except the actual validation set) is used for training. So,
    # there is no data left for last-layer retraining, and we return.
    if args.split == "combined" and args.train_pct == 100:
        return

    # Gets last-epoch ERM weights.
    args.weights = get_weights(erm_version, ind=-1)

    # Sets finetune types. Note that "group-unbalanced retraining" will be
    # either class-unbalanced or class-balanced based on the value
    # of args.balanced_sampler.
    finetune_types = [
        "group-unbalanced retraining",
        "group-balanced retraining",
        "misclassification finetuning",
        "early-stop misclassification finetuning",
        "dropout disagreement finetuning",
        "early-stop disagreement finetuning",
    ]

    # Prepares results dictionary.
    results = {f: {"val_worst_group_acc": -1, "metrics": [], "params": []}
               for f in finetune_types}

    def finetune_helper(
        finetune_type,
        dropout_prob=0,
        early_stop_num=None,
        finetune_lr=None,
        finetune_num_data=None,
        worst_group_pct=None,
    ):
        args.finetune_lr = finetune_lr if finetune_lr else args.lr

        param_str = f" LR {args.finetune_lr}"
        if finetune_num_data:
            param_str += f" Num Data {finetune_num_data}"
        if dropout_prob:
            param_str += f" Dropout {dropout_prob}"
        if early_stop_num:
            param_str += f" Early Stop {early_stop_num}"
        if worst_group_pct:
            param_str += f" Worst-group Pct {worst_group_pct}"
        print(f"{finetune_type.title()}{param_str}")

        early_stop_weights = None
        if early_stop_num:
            early_stop_weights = get_weights(erm_version, ind=early_stop_num-1)
        val_metrics, test_metrics = finetune_last_layer(
            args,
            finetune_type,
            model_class,
            dropout_prob=dropout_prob,
            early_stop_weights=early_stop_weights,
            finetune_num_data=finetune_num_data,
            worst_group_pct=worst_group_pct,
        )

        print_metrics([val_metrics, test_metrics], test_group_proportions=args.test_group_proportions)

        # Hack for CIFAR-10
        if "test_acc1" in test_metrics[0].keys():
            sys.exit(0)

        val_worst_group_acc = min([group[f"val_acc1/dataloader_idx_{j+1}"]
                                   for j, group in enumerate(val_metrics[1:])])

        if val_worst_group_acc > results[finetune_type]["val_worst_group_acc"]:
            results[finetune_type]["val_worst_group_acc"] = val_worst_group_acc
            results[finetune_type]["metrics"] = [val_metrics, test_metrics]

            params = []
            if finetune_num_data:
                params.append(finetune_num_data)
            if finetune_lr:
                params.append(finetune_lr)
            if dropout_prob:
                params.append(dropout_prob)
            if early_stop_num:
                params.append(early_stop_num)
            results[finetune_type]["params"] = params

    # Performs worst-group data ablation.
    if args.worst_group_ablation:
        for worst_group_pct in [2.5, 5] + [12.5 * j for j in range(1, 9)]:
            finetune_helper("class-balanced retraining", worst_group_pct=worst_group_pct)
        return

    # Performs early-stop disagreement demo.
    if args.demo:
        finetune_helper(
            "early-stop disagreement finetuning",
            early_stop_num=1,
            finetune_lr=1e-2,
            finetune_num_data=100,
        )
        print_results2(results, finetune_types[-1:])
        return

    # Performs last-layer retraining.
    finetune_helper("group-unbalanced retraining")
    finetune_helper("group-balanced retraining")

    if args.no_self:
        print_results2(results, finetune_types[:2])
        return

    # Perform SELF hyperparameter search using worst-group validation accuracy.
    for finetune_num_data in args.finetune_num_datas:
        for finetune_lr in args.finetune_lrs:
            def finetune_helper2(finetune_type, dropout_prob=0, early_stop_num=None):
                finetune_helper(
                    finetune_type,
                    dropout_prob=dropout_prob,
                    early_stop_num=early_stop_num,
                    finetune_lr=finetune_lr,
                    finetune_num_data=finetune_num_data,
                )

            finetune_helper2("misclassification finetuning")

            for early_stop_num in args.early_stop_nums:
                finetune_helper2("early-stop misclassification finetuning", early_stop_num=early_stop_num)

            for dropout_prob in args.dropout_probs:
                finetune_helper2("dropout disagreement finetuning", dropout_prob=dropout_prob)

            for early_stop_num in args.early_stop_nums:
                finetune_helper2("early-stop disagreement finetuning", early_stop_num=early_stop_num)

    print_results2(results, finetune_types)

    
if __name__ == "__main__":
    args = parse_args()

    models, _ = valid_models_and_datamodules()
        
    experiment(args, models[args.model])

