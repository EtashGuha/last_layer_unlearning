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


def reset_fc(model):
    for layer in model.model.fc:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

def save_metrics(cfg, val, params, metrics, name):
    with open("disagreement.pkl", "rb") as f:
        save_state = pickle.load(f)
        save_state[cfg][f"{name}_best_worst_group_val"] = val
        save_state[cfg][f"{name}_set_params"] = params
        save_state[cfg][f"{name}_set_metrics"] = metrics
    with open("disagreement.pkl", "wb") as f:
        pickle.dump(save_state, f)

def save_idx(cfg, j):
    with open("disagreement.pkl", "rb") as f:
        save_state = pickle.load(f)
    save_state[cfg]["start_class_weight_idx"] = j
    with open("disagreement.pkl", "wb") as f:
        pickle.dump(save_state, f)

def disagreement(
    args,
    gamma=None,
    misclassification_dfr=False,
    orig_dfr=False,
    dropout=0,
    rebalancing=True,
    class_weights=[1., 1.],
    dfr_epochs=100,
    disagreement_ablation=False,
    kldiv_proportion=None,
):
    disagreement_args = deepcopy(args)
    disagreement_args.dropout_prob = dropout
    disagreement_args.balanced_sampler = True if (rebalancing and not orig_dfr) else False #orig_dfr uses balancing by definition

    finetune_args = deepcopy(args)
    finetune_args.train_fc_only = True
    finetune_args.check_val_every_n_epoch = dfr_epochs
    finetune_args.ckpt_every_n_epochs = dfr_epochs
    finetune_args.max_epochs = dfr_epochs
    finetune_args.class_weights = class_weights
    if args.finetune_weights:
        finetune_args.weights = args.finetune_weights

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
                    dropout_dfr=(dropout > 0),
                    disagreement_ablation=disagreement_ablation,
                    kldiv_proportion=kldiv_proportion,
                )

        _, val_metrics, test_metrics = main(finetune_args, BERT, CivilCommentsDisagreement2, model_hooks=[reset_fc])

    return val_metrics, test_metrics

def experiment(args):
    # Instantiates save state.
    if not osp.isfile("disagreement.pkl"):
        with open("disagreement.pkl", "wb") as f:
            pickle.dump({}, f)

    # Loads save state.
    with open("disagreement.pkl", "rb") as f:
        save_state = pickle.load(f)
        cfg = f"{args.seed}{args.datamodule}{args.disagreement_proportion}{args.disagreement_from_early_stop_epochs}"
        base_model_cfg = f"{args.seed}{args.datamodule}"

        resume = None
        if cfg in save_state:
            resume = save_state[cfg]
        else:
            save_state[cfg] = {}

        base_model_resume = None
        if base_model_cfg in save_state:
            base_model_resume = save_state[base_model_cfg]
        else:
            save_state[base_model_cfg] = {}
    with open("disagreement.pkl", "wb") as f:
        pickle.dump(save_state, f)

    # Hyperparameter search specifications
    #CLASS_WEIGHTS = [[1., 1.], [1., 2.], [1., 5.]]
    CLASS_WEIGHTS = [[1., 1.]]
    PROPORTIONS = [0.005, 0.025, 0.05, 0.125, 0.25]
    DROPOUTS = [0.1, 0.3, 0.5, 0.7, 0.9]

    if args.datamodule == "waterbirds":
        dm = WaterbirdsDisagreement
    elif args.datamodule == "celeba":
        dm = CelebADisagreement
    elif args.datamodule == "civilcomments":
        dm = CivilCommentsDisagreement
    args.num_classes = 2

    # full epochs model
    if base_model_resume and "erm_version" in base_model_resume and "erm_metrics" in base_model_resume:
        version = base_model_resume["erm_version"]
        erm_metrics = base_model_resume["erm_metrics"]
    else:
        # resume if interrupted (need to manually add version)
        if base_model_resume and "erm_version" in base_model_resume:
            version = base_model_resume["erm_version"]
            list_of_weights = glob(osp.join(os.getcwd(), f"lightning_logs/version_{version}/checkpoints/*"))
            args.weights = max(list_of_weights, key=os.path.getctime)
            args.resume_training = True
        model, erm_val_metrics, erm_test_metrics = main(args, ResNet, dm)
        if not args.resume_training:
            version = model.trainer.logger.version
        erm_metrics = [erm_val_metrics, erm_test_metrics]
        args.weights = ""
        args.resume_training = ""
        del model

        with open("disagreement.pkl", "rb") as f:
            save_state = pickle.load(f)
        save_state[base_model_cfg]["erm_version"] = version
        save_state[base_model_cfg]["erm_metrics"] = erm_metrics
        with open("disagreement.pkl", "wb") as f:
            pickle.dump(save_state, f)
 
    # e.g., disagree from 10 epochs but finetune from 100
    args.finetune_weights = None
    if args.disagreement_from_early_stop_epochs:
        # TODO: CHANGE THIS.
        args.finetune_weights = osp.join(os.getcwd(), f"lightning_logs/version_{version}/checkpoints/last.ckpt")
        args.max_epochs = args.disagreement_from_early_stop_epochs
        args.check_val_every_n_epoch = args.max_epochs

        if resume and "early_version" in resume:
            version = resume["early_version"]
        else:
            model, _, _ = main(args, ResNet, dm)
            version = model.trainer.logger.version
            del model

            with open("disagreement.pkl", "rb") as f:
                save_state = pickle.load(f)
            save_state[cfg]["early_version"] = version
            with open("disagreement.pkl", "wb") as f:
                pickle.dump(save_state, f)

    list_of_weights = glob(osp.join(os.getcwd(), f"lightning_logs/version_{version}/checkpoints/*"))
    args.weights = max(list_of_weights, key=os.path.getctime)

    # for testing
    #val_metrics, test_metrics = disagreement(args, gamma=1, dropout=0.3, class_weights=[1., 1.])
    #print(test_metrics)
    #return

    # Loads current hyperparam search cfg if needed.
    orig_val = 0
    miscls_val = 0
    dropout_val = 0
    start_class_weight_idx = 0
    if resume and "orig_val" in resume and "orig_params" in resume and "orig_metrics" in resume:
        orig_val = resume["orig_val"]
        orig_params = resume["orig_params"]
        orig_metrics = resume["orig_metrics"]
    if resume and "miscls_val" in resume and "miscls_params" in resume and "miscls_metrics" in resume:
        miscls_val = resume["miscls_val"]
        miscls_params = resume["miscls_params"]
        miscls_metrics = resume["miscls_metrics"]
    if resume and "dropout_val" in resume and "dropout_params" in resume and "dropout_metrics" in resume:
        dropout_val = resume["dropout_val"]
        dropout_params = resume["dropout_params"]
        dropout_metrics = resume["dropout_metrics"]
    if resume and "start_class_weight_idx" in resume:
        start_class_weight_idx = resume["start_class_weight_idx"]

    # Does hyperparameter search based on worst group validation error.
    for j, class_weights in enumerate(CLASS_WEIGHTS):
        # Skips to the right point if resuming.
        if j < start_class_weight_idx:
            continue
        save_idx(cfg, j)

        print(f"Full Set DFR: Class Weights {class_weights}")
        val_metrics, test_metrics = disagreement(args, orig_dfr=True, class_weights=class_weights)

        best_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
        if best_wg_val > orig_val:
            orig_val = best_wg_val
            orig_params = [class_weights]
            orig_metrics = [val_metrics, test_metrics]

            save_metrics(cfg, orig_val, orig_params, orig_metrics, "orig")

        print(f"Misclassification DFR: Class Weights {class_weights} Gamma 1")
        val_metrics, test_metrics = disagreement(args, gamma=1, misclassification_dfr=True, class_weights=class_weights)

        best_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
        if best_wg_val > miscls_val:
            miscls_val = best_wg_val
            miscls_params = [class_weights, 1]
            miscls_metrics = [val_metrics, test_metrics]

            save_metrics(cfg, miscls_val, miscls_params, miscls_metrics, "miscls")

        for proportion in PROPORTIONS:
            for dropout in DROPOUTS:
                print(f"Dropout DFR: Class Weights {class_weights} Proportion {proportion} Dropout {dropout}")
                val_metrics, test_metrics = disagreement(args, kldiv_proportion=proportion, dropout=dropout, class_weights=class_weights)

                best_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
                if best_wg_val > dropout_val:
                    dropout_val = best_wg_val
                    dropout_params = [class_weights, proportion, dropout]
                    dropout_metrics = [val_metrics, test_metrics]

                    save_metrics(cfg, dropout_val, dropout_params, dropout_metrics, "dropout")

    save_idx(cfg, len(CLASS_WEIGHTS))
    class_weights, proportion, dropout = dropout_params
    print("Rebalancing Ablation")
    _, rebalancing_ablation_metrics = disagreement(args, kldiv_proportion=proportion, dropout=dropout, class_weights=class_weights, rebalancing=False)
    print("KLDiv Bottom Ablation")
    _, bottom_ablation_metrics = disagreement(args, kldiv_top_proportion=proportion, kldiv_bottom_proportion=0, dropout=dropout, class_weights=class_weights)
    print("KLDiv Top Ablation")
    _, top_ablation_metrics = disagreement(args, kldiv_top_proportion=0, kldiv_bottom_proportion=proportion, dropout=dropout, class_weights=class_weights)

    print("\n---Hyperparameter Search Results---")
    print("\nERM:")
    print(erm_metrics)
    print("\nFull Set DFR:")
    print(full_set_params)
    print(full_set_metrics)
    print("\nMisclassification DFR:")
    print(misclassification_params)
    print(misclassification_metrics)
    print("\nDropout DFR:")
    print(dropout_params)
    print(dropout_metrics)
    print("\nRebalancing Ablation w/ Best Dropout Config")
    print(rebalancing_ablation_metrics)
    print("\nKLDiv Bottom Ablation w/ Best Dropout Config")
    print(bottom_ablation_metrics)
    print("\nKLDiv Top Ablation w/ Best Dropout Config")
    print(top_ablation_metrics)

if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add("--disagreement_proportion", type=float)
    parser.add("--disagreement_from_early_stop_epochs", default=0, type=int)

    args = parser.parse_args()

    experiment(args)

