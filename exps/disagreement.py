"""Main file for Dropout DFR experimentation."""

# Imports Python builtins.
from configargparse import Parser
from copy import deepcopy
from glob import glob
import os
import os.path as osp
import pickle

# Imports PyTorch packages.
from pytorch_lightning import Trainer

# Imports groundzero packages.
from groundzero.args import add_input_args
from groundzero.datamodules.celeba import CelebADisagreement
from groundzero.datamodules.civilcomments import CivilComments
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.main import load_model, main
from groundzero.models.bert import BERT
from groundzero.models.resnet import ResNet


def get_latest_weights(version):
    list_of_weights = glob(osp.join(os.getcwd(), f"lightning_logs/version_{version}/checkpoints/*"))
    return max(list_of_weights, key=os.path.getctime)

def load_save_state(args):
    # Instantiates save state.
    if not osp.isfile("disagreement.pkl"):
        with open("disagreement.pkl", "wb") as f:
            pickle.dump({}, f)

    # Loads save state.
    with open("disagreement.pkl", "rb") as f:
        save_state = pickle.load(f)
        cfg = f"{args.seed}{args.datamodule}{args.disagreement_proportion}"
        erm_cfg = f"{args.seed}{args.datamodule}"

        resume = None
        if cfg in save_state:
            resume = save_state[cfg]
        else:
            save_state[cfg] = {}

        erm_resume = None
        if erm_cfg in save_state:
            erm_resume = save_state[erm_cfg]
        else:
            save_state[erm_cfg] = {}
    with open("disagreement.pkl", "wb") as f:
        pickle.dump(save_state, f)
        
    return cfg, erm_cfg, resume, erm_resume

def reset_fc(model):
    for layer in model.model.fc:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

def save_metrics(cfg, metrics, names):
    # Must pass either 2 lists/tuples of the same length or 2 singletons..
    if (type(metrics) == list or type(metrics) == tuple
        or type(names) == list or type(names) == tuple):
        assert type(metrics) == type(names)
        assert len(metrics) == len(names)
    else:
        metrics = (metrics,)
        names = (names,)

    with open("disagreement.pkl", "rb") as f:
        save_state = pickle.load(f)
        for metric, name in zip(metrics, names):
            save_state[cfg][name] = metric
    with open("disagreement.pkl", "wb") as f:
        pickle.dump(save_state, f)

def print_metrics(test_metrics, train_dist_proportion):
    train_dist_mean_acc = sum([p * group[f"test_acc1/dataloader_idx_{j+1}"] for j, (group, p) in enumerate(zip(test_metrics[1:], train_dist_proportion))])
    test_dist_mean_acc = test_metrics[0]["test_acc1/dataloader_idx_0"]
    worst_group_acc = min([group[f"test_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(test_metrics[1:])])

    train_dist_mean_acc = round(train_dist_mean_acc * 100, 1)
    test_dist_mean_acc = round(test_dist_mean_acc * 100, 1)
    worst_group_acc = round(worst_group_acc * 100, 1)

    print(f"Train Dist Mean Acc: {train_dist_mean_acc}")
    print(f"Test Dist Mean Acc: {test_dist_mean_acc}")
    print(f"Worst Group Acc: {worst_group_acc}")

def disagreement(
    args,
    gamma=None,
    misclassification_dfr=False,
    orig_dfr=False,
    random_dfr=False,
    dropout=0,
    rebalancing=True,
    class_weights=[1., 1.],
    dfr_epochs=100,
    disagreement_ablation=False,
    kldiv_proportion=None,
):
    disagreement_args = deepcopy(args)
    disagreement_args.dropout_prob = dropout
    #disagreement_args.balanced_sampler = True if (rebalancing and not orig_dfr) else False #orig_dfr uses balancing by definition
    disagreement_args.balanced_sampler = False # using all class labels to condition on classes beforehand

    finetune_args = deepcopy(args)
    finetune_args.train_fc_only = True
    finetune_args.check_val_every_n_epoch = dfr_epochs
    finetune_args.ckpt_every_n_epochs = dfr_epochs
    finetune_args.max_epochs = dfr_epochs
    finetune_args.class_weights = class_weights

    if args.datamodule == "waterbirds":
        model = load_model(disagreement_args, ResNet)

        class WaterbirdsDisagreement2(WaterbirdsDisagreement):
            def __init__(self, args):
                super().__init__(
                    disagreement_args,
                    model=model,
                    gamma=gamma,
                    orig_dfr=orig_dfr,
                    misclassification_dfr=misclassification_dfr,
                    random_dfr=random_dfr,
                    dropout_dfr=(dropout > 0),
                    disagreement_ablation=disagreement_ablation,
                    kldiv_proportion=kldiv_proportion,
                )

        _, val_metrics, test_metrics = main(finetune_args, ResNet, WaterbirdsDisagreement2, model_hooks=[reset_fc])
    elif args.datamodule == "celeba":
        model = load_model(disagreement_args, ResNet)

        class CelebADisagreement2(CelebADisagreement):
            def __init__(self, args):
                super().__init__(
                    disagreement_args,
                    model=model,
                    gamma=gamma,
                    orig_dfr=orig_dfr,
                    misclassification_dfr=misclassification_dfr,
                    random_dfr=random_dfr,
                    dropout_dfr=(dropout > 0),
                    disagreement_ablation=disagreement_ablation,
                    kldiv_proportion=kldiv_proportion,
                )

        _, val_metrics, test_metrics = main(finetune_args, ResNet, CelebADisagreement2, model_hooks=[reset_fc])
    elif args.datamodule == "civilcomments":
        model = load_model(disagreement_args, BERT)

        class CivilCommentsDisagreement2(CivilCommentsDisagreement):
            def __init__(self, args):
                super().__init__(
                    disagreement_args,
                    model=model,
                    gamma=gamma,
                    orig_dfr=orig_dfr,
                    misclassification_dfr=misclassification_dfr,
                    random_dfr=random_dfr,
                    dropout_dfr=(dropout > 0),
                    disagreement_ablation=disagreement_ablation,
                    kldiv_proportion=kldiv_proportion,
                )

        _, val_metrics, test_metrics = main(finetune_args, BERT, CivilCommentsDisagreement2, model_hooks=[reset_fc])

    return val_metrics, test_metrics

def experiment(args):
    # Loads save state from pickle file.
    cfg, erm_cfg, resume, erm_resume = load_save_state(args)

    # Sets search parameters.
    PROPORTIONS = [1, 2, 5, 10, 20, 50]
    DROPOUTS = [0.5, 0.7, 0.9]
    
    # Sets datamodule-specific parameters.
    if args.datamodule == "waterbirds":
        dm = WaterbirdsDisagreement
        args.num_classes = 2
        CLASS_WEIGHTS = [[1., 1.]]
        TRAIN_DIST_PROPORTION = [0.7295, 0.0384, 0.2204, 0.0117]
    elif args.datamodule == "celeba":
        dm = CelebADisagreement
        args.num_classes = 2
        CLASS_WEIGHTS = [[1., 2.], [1., 3.], [1., 5.]]
        TRAIN_DIST_PROPORTION = [0.4295, 0.4166, 0.1447, 0.0092]
    elif args.datamodule == "civilcomments":
        dm = CivilCommentsDisagreement
        args.num_classes = 2
        CLASS_WEIGHTS = [[1., 1.]]
        TRAIN_DIST_PROPORTION = []

    try:
        # Loads ERM model.
        erm_version = erm_resume["version"]
        erm_metrics = erm_resume["metrics"]
    except:
        # Resumes ERM training if interrupted (need to manually add version).
        if erm_resume:
            erm_version = erm_resume["version"]
            args.weights = get_latest_weights(erm_version)
            args.resume_training = True
        model, erm_val_metrics, erm_test_metrics = main(args, ResNet, dm)
        erm_version = model.trainer.logger.version
        erm_metrics = [erm_val_metrics, erm_test_metrics]
        args.weights = ""
        args.resume_training = False
        del model

        save_metrics(erm_cfg, (erm_version, erm_metrics), ("version", "metrics"))
 
    args.weights = get_latest_weights(erm_version)

    # Loads current hyperparam search cfg if needed.
    orig = {"val": 0}
    miscls = {"val": 0}
    random = {proportion: {"val": 0} for proportion in PROPORTIONS}
    dropout = {proportion: {"val": 0} for proportion in PROPORTIONS}
    start_idx = 0
    if resume:
        if "orig" in resume:
            orig = resume["orig"]
        if "miscls" in resume:
            miscls = resume["miscls"]
        if "random" in resume:
            random = resume["random"]
        if "dropout" in resume:
            dropout = resume["dropout"]
        if "start_idx" in resume:
            start_idx = resume["start_idx"]

    # Does hyperparameter search based on worst group validation error.
    for j, class_weights in enumerate(CLASS_WEIGHTS):
        # Skips to the right point if resuming.
        if j < start_idx:
            continue
        save_metrics(cfg, j, "start_idx")

        print(f"Original DFR: Class Weights {class_weights}")
        val_metrics, test_metrics = disagreement(args, orig_dfr=True, class_weights=class_weights)

        best_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
        if best_wg_val > orig["val"]:
            orig["val"] = best_wg_val
            orig["params"] = [class_weights]
            orig["metrics"] = [val_metrics, test_metrics]

            save_metrics(cfg, orig, "orig")

        print(f"Misclassification DFR: Class Weights {class_weights}")
        val_metrics, test_metrics = disagreement(args, gamma=1, misclassification_dfr=True, class_weights=class_weights)

        best_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
        if best_wg_val > miscls["val"]:
            miscls["val"] = best_wg_val
            miscls["params"] = [class_weights]
            miscls["metrics"] = [val_metrics, test_metrics]

            save_metrics(cfg, miscls, "miscls")

        for proportion in PROPORTIONS:
            print(f"Random DFR Proportion {proportion}: Class Weights {class_weights}")
            val_metrics, test_metrics = disagreement(args, kldiv_proportion=proportion/200, class_weights=class_weights, random_dfr=True)

            best_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
            if best_wg_val > random[proportion]["val"]:
                random[proportion]["val"] = best_wg_val
                random[proportion]["params"] = [class_weights]
                random[proportion]["metrics"] = [val_metrics, test_metrics]

                save_metrics(cfg, random, "random")

            for drop in DROPOUTS:
                print(f"Dropout DFR Proportion {proportion}: Class Weights {class_weights} Dropout {drop}")
                val_metrics, test_metrics = disagreement(args, kldiv_proportion=proportion/200, dropout=drop, class_weights=class_weights)

                best_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
                if best_wg_val > dropout[proportion]["val"]:
                    dropout[proportion]["val"] = best_wg_val
                    dropout[proportion]["params"] = [class_weights, drop]
                    dropout[proportion]["metrics"] = [val_metrics, test_metrics]

                    save_metrics(cfg, dropout, "dropout")

    save_metrics(cfg, len(CLASS_WEIGHTS), "start_idx")

    """
    class_weights, proportion, dropout = dropout_params
    print("Rebalancing Ablation")
    _, rebalancing_ablation_metrics = disagreement(args, kldiv_proportion=proportion, dropout=dropout, class_weights=class_weights, rebalancing=False)
    print("KLDiv Bottom Ablation")
    _, bottom_ablation_metrics = disagreement(args, kldiv_top_proportion=proportion, kldiv_bottom_proportion=0, dropout=dropout, class_weights=class_weights)
    print("KLDiv Top Ablation")
    _, top_ablation_metrics = disagreement(args, kldiv_top_proportion=0, kldiv_bottom_proportion=proportion, dropout=dropout, class_weights=class_weights)
    """

    print("\n---Hyperparameter Search Results---")
    print("\nERM:")
    print_metrics(erm_metrics[1], TRAIN_DIST_PROPORTION)
    print("\nOriginal DFR:")
    print(orig["params"])
    print_metrics(orig["metrics"][1], TRAIN_DIST_PROPORTION)
    print("\nMisclassification DFR:")
    print(miscls["params"])
    print_metrics(miscls["metrics"][1], TRAIN_DIST_PROPORTION)
    for proportion in PROPORTIONS:
        print(f"\nRandom DFR Proportion {proportion}:")
        print(random[proportion]["params"])
        print_metrics(random[proportion]["metrics"][1], TRAIN_DIST_PROPORTION)
        print(f"\nDropout DFR Proportion {proportion}:")
        print(dropout[proportion]["params"])
        print_metrics(dropout[proportion]["metrics"][1], TRAIN_DIST_PROPORTION)
    
    """
    print("\nRebalancing Ablation w/ Best Dropout Config")
    print(rebalancing_ablation_metrics)
    print("\nKLDiv Bottom Ablation w/ Best Dropout Config")
    print(bottom_ablation_metrics)
    print("\nKLDiv Top Ablation w/ Best Dropout Config")
    print(top_ablation_metrics)
    """

if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add("--disagreement_proportion", type=float)
    #parser.add("--disagreement_from_early_stop_epochs", default=0, type=int)

    args = parser.parse_args()

    experiment(args)

