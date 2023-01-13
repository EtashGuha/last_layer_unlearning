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
from groundzero.datamodules.civilcomments import CivilCommentsDisagreement
from groundzero.datamodules.fmow import FMOWDisagreement
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.main import load_model, main
from groundzero.models.bert import BERT
from groundzero.models.resnet import ResNet


def get_latest_weights(version):
    list_of_weights = glob(osp.join(os.getcwd(), f"lightning_logs/version_{version}/checkpoints/*"))
    return max(list_of_weights, key=osp.getctime)

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
    try:
        for layer in model.model.fc:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    except:
        model.model.fc.reset_parameters()

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

def dfr(
    args,
    dfr_type,
    class_labels_proportion,
    class_balancing=True,
    class_weights=[],
    dfr_epochs=100,
    dropout_prob=None,
    kl_ablation=None,
):
    disagreement_args = deepcopy(args)
    disagreement_args.dfr_type = dfr_type
    disagreement_args.dropout_prob = dropout_prob

    finetune_args = deepcopy(args)
    finetune_args.train_fc_only = True
    finetune_args.check_val_every_n_epoch = dfr_epochs
    finetune_args.ckpt_every_n_epochs = dfr_epochs
    finetune_args.max_epochs = dfr_epochs
    finetune_args.class_weights = class_weights
    finetune_args.lr = 1e-3
    finetune_args.lr_scheduler = "step"
    finetune_args.lr_steps = []

    def disagreement_cls(orig_cls):
        class Disagreement(orig_cls):
            def __init__(self, args):
                super().__init__(
                    disagreement_args,
                    model=model,
                    class_balancing=class_balancing,
                    class_labels_proportion=class_labels_proportion,
                    kl_ablation=kl_ablation,
                )

        return Disagreement

    if args.datamodule == "waterbirds":
        model = load_model(disagreement_args, ResNet)
        _, val_metrics, test_metrics = main(finetune_args, ResNet, disagreement_cls(WaterbirdsDisagreement), model_hooks=[reset_fc])
    elif args.datamodule == "celeba":
        model = load_model(disagreement_args, ResNet)
        _, val_metrics, test_metrics = main(finetune_args, ResNet, disagreement_cls(CelebADisagreement), model_hooks=[reset_fc])
    elif args.datamodule == "fmow":
        model = load_model(disagreement_args, ResNet)
        _, val_metrics, test_metrics = main(finetune_args, ResNet, disagreement_cls(FMOWDisagreement), model_hooks=[reset_fc])
    elif args.datamodule == "civilcomments":
        model = load_model(disagreement_args, BERT)
        _, val_metrics, test_metrics = main(finetune_args, BERT, disagreement_cls(CivilCommentsDisagreement), model_hooks=[reset_fc])
    else:
        raise ValueError("DataModule not supported.")

    return val_metrics, test_metrics

def experiment(args):
    # Loads save state from pickle file.
    cfg, erm_cfg, resume, erm_resume = load_save_state(args)

    # Sets search parameters.
    PROPORTIONS = [1, 2, 5, 10, 20, 50]
    DROPOUT_PROBS = [0.5, 0.7, 0.9]
    
    # Sets datamodule-specific parameters.
    if args.datamodule == "waterbirds":
        model_cls = ResNet
        dm = WaterbirdsDisagreement
        dfr_epochs = 100
        args.num_classes = 2
        CLASS_WEIGHTS = [[]]
        TRAIN_DIST_PROPORTION = [0.7295, 0.0384, 0.2204, 0.0117]
    elif args.datamodule == "celeba":
        model_cls = ResNet
        dm = CelebADisagreement
        dfr_epochs = 100
        args.num_classes = 2
        CLASS_WEIGHTS = [[1., 2.], [1., 3.], [1., 5.]]
        TRAIN_DIST_PROPORTION = [0.4295, 0.4166, 0.1447, 0.0092]
    elif args.datamodule == "fmow":
        model_cls = ResNet
        dm = FMOWDisagreement
        dfr_epochs = 100
        args.num_classes = 62
        CLASS_WEIGHTS = [[]]
        TRAIN_DIST_PROPORTION = [0.2318, 0.4532, 0.0206, 0.2730, 0.0214]
    elif args.datamodule == "civilcomments":
        model_cls = BERT
        dm = CivilCommentsDisagreement
        dfr_epochs = 20
        args.num_classes = 2
        CLASS_WEIGHTS = [[]]
        TRAIN_DIST_PROPORTION = [0.5508, 0.3358, 0.0473, 0.0661]

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
        model, erm_val_metrics, erm_test_metrics = main(args, model_cls, dm)
        erm_version = model.trainer.logger.version
        erm_metrics = [erm_val_metrics, erm_test_metrics]
        args.weights = ""
        args.resume_training = False
        del model

        save_metrics(erm_cfg, (erm_version, erm_metrics), ("version", "metrics"))
 
    args.weights = get_latest_weights(erm_version)

    # Loads current hyperparam search cfg if needed.
    orig = {proportion: {label: {"val": -1} for label in [True, False]} for proportion in PROPORTIONS}
    random = deepcopy(orig)
    miscls = deepcopy(orig)
    dropout = deepcopy(orig)
    orig.update({100: {label: {"val": -1} for label in [True, False]}})
    random.update({100: {label: {"val": -1} for label in [True, False]}})
    if resume:
        orig = resume["orig"] if "orig" in resume else orig
        random = resume["random"] if "random" in resume else random
        miscls = resume["miscls"] if "miscls" in resume else miscls
        dropout = resume["dropout"] if "dropout" in resume else dropout

    results = {"orig": orig, "random": random, "miscls": miscls, "dropout": dropout}
    def dfr_helper(
        args,
        dfr_type,
        class_labels_proportion,
        class_balancing=True,
        class_weights=[],
        dropout_prob=0,
        kl_ablation=None,
    ):
        val_metrics, test_metrics = dfr(
            args,
            dfr_type,
            class_labels_proportion / 100,
            class_balancing=class_balancing,
            class_weights=class_weights,
            dfr_epochs=dfr_epochs,
            dropout_prob=dropout_prob,
            kl_ablation=kl_ablation,
        )

        worst_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
        if worst_wg_val >= results[dfr_type][class_labels_proportion][class_balancing]["val"]:
            results[dfr_type][class_labels_proportion][class_balancing]["val"] = worst_wg_val
            params = [class_weights, dropout_prob] if dropout_prob else [class_weights]
            results[dfr_type][class_labels_proportion][class_balancing]["params"] = params
            results[dfr_type][class_labels_proportion][class_balancing]["metrics"] = [val_metrics, test_metrics]
            save_metrics(cfg, results[dfr_type], dfr_type)

    # Does hyperparameter search based on worst group validation error.
    for class_weights in CLASS_WEIGHTS:
        print(f"Group-Balanced DFR Proportion 100: Class Weights {class_weights}")
        dfr_helper(args, "orig", 100, class_weights=class_weights)

        print(f"Class-Balanced DFR Proportion 100: Class Weights {class_weights}")
        dfr_helper(args, "random", 100, class_weights=class_weights)

        print(f"Unbalanced DFR Proportion 100: Class Weights {class_weights}")
        dfr_helper(args, "random", 100, class_balancing=False, class_weights=class_weights) 

        for proportion in PROPORTIONS:
            print(f"Group-Balanced DFR Proportion {proportion}: Class Weights {class_weights}")
            dfr_helper(args, "orig", proportion, class_weights=class_weights)

            print(f"Random DFR Proportion {proportion}: Class Weights {class_weights}")
            dfr_helper(args, "random", proportion, class_weights=class_weights)

            print(f"Misclassification DFR Proportion {proportion}: Class Weights {class_weights}")
            dfr_helper(args, "miscls", proportion, class_weights=class_weights)

            for dropout_prob in DROPOUT_PROBS:
                print(f"Dropout DFR Proportion {proportion}: Class Weights {class_weights} Dropout {dropout_prob}")
                dfr_helper(args, "dropout", proportion, class_weights=class_weights, dropout_prob=dropout_prob)

                """
                print(f"Dropout DFR Proportion {proportion}: Class Weights {class_weights} Dropout {dropout_prob} TOP")
                dfr_helper(args, "dropout", proportion, class_weights=class_weights, dropout_prob=dropout_prob, kl_ablation="top")

                print(f"Dropout DFR Proportion {proportion}: Class Weights {class_weights} Dropout {dropout_prob} RANDOM")
                dfr_helper(args, "dropout", proportion, class_weights=class_weights, dropout_prob=dropout_prob, kl_ablation="random")
                """

    # TODO: Add rebalancing ablations back in.

    print("\n---Hyperparameter Search Results---")
    print("\nERM:")
    print_metrics(erm_metrics[1], TRAIN_DIST_PROPORTION)
    print("\nGroup-Balanced DFR Proportion 100:")
    print(orig[100][True]["params"])
    print_metrics(orig[100][True]["metrics"][1], TRAIN_DIST_PROPORTION)
    print("\nClass-Balanced DFR Proportion 100:")
    print(random[100][True]["params"])
    print_metrics(random[100][True]["metrics"][1], TRAIN_DIST_PROPORTION)
    print("\nUnbalanced DFR Proportion 100:")
    print(random[100][False]["params"])
    print_metrics(random[100][False]["metrics"][1], TRAIN_DIST_PROPORTION)

    for proportion in PROPORTIONS:
        print(f"\nGroup-Balanced DFR Proportion {proportion}:")
        print(orig[proportion][True]["params"])
        print_metrics(orig[proportion][True]["metrics"][1], TRAIN_DIST_PROPORTION)
        print(f"\nRandom DFR Proportion {proportion}:")
        print(random[proportion][True]["params"])
        print_metrics(random[proportion][True]["metrics"][1], TRAIN_DIST_PROPORTION)
        print(f"\nMisclassification DFR Proportion {proportion}:")
        print(miscls[proportion][True]["params"])
        print_metrics(miscls[proportion][True]["metrics"][1], TRAIN_DIST_PROPORTION)
        print(f"\nDropout DFR Proportion {proportion}:")
        print(dropout[proportion][True]["params"])
        print_metrics(dropout[proportion][True]["metrics"][1], TRAIN_DIST_PROPORTION)
    
if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # The proportion of the validation set to use for DFR. The remainder
    # of the validation set will be used for model selection.
    parser.add("--disagreement_proportion", type=float, default=0.5)

    args = parser.parse_args()

    experiment(args)

