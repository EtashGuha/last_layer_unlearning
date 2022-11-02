from configargparse import Parser
from copy import deepcopy
from glob import glob
import os
import os.path as osp
import pickle

from pytorch_lightning import Trainer

from groundzero.args import add_input_args
from groundzero.datamodules.celeba import CelebADisagreement
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.main import load_model, main
from groundzero.models.resnet import ResNet


def disagreement(args, gamma=1, misclassification_dfr=False, orig_dfr=False, dropout=0, rebalancing=True, class_weights=[1., 1.], dfr_epochs=100, disagreement_ablation=False):
    disagreement_args = deepcopy(args)
    disagreement_args.dropout_prob = dropout
    model = load_model(disagreement_args, ResNet)

    finetune_args = deepcopy(args)
    finetune_args.train_fc_only = True
    finetune_args.check_val_every_n_epoch = dfr_epochs
    finetune_args.ckpt_every_n_epochs = dfr_epochs
    finetune_args.max_epochs = dfr_epochs
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
                    orig_dfr=orig_dfr,
                    misclassification_dfr=misclassification_dfr,
                    dropout_dfr=(dropout > 0),
                    rebalancing=rebalancing,
                    disagreement_ablation=disagreement_ablation,
                )

        _, val_metrics, test_metrics = main(finetune_args, ResNet, WaterbirdsDisagreement2, reset_fc=True)
    elif args.datamodule == "celeba":
        class CelebADisagreement2(CelebADisagreement):
            def __init__(self, args):
                super().__init__(
                    disagreement_args,
                    model=model,
                    gamma=gamma,
                    orig_dfr=orig_dfr,
                    misclassification_dfr=misclassification_dfr,
                    dropout_dfr=(dropout > 0),
                    rebalancing=rebalancing,
                    disagreement_ablation=disagreement_ablation,
                )

        _, val_metrics, test_metrics = main(finetune_args, ResNet, CelebADisagreement2, reset_fc=True)

    return val_metrics, test_metrics

def experiment(args):
    # instantiate save state
    if not osp.isfile("disagreement.pkl"):
        x = {}
        with open("disagreement.pkl", "wb") as f:
            pickle.dump(x, f)

    # load save state
    with open("disagreement.pkl", "rb") as f:
        save_state = pickle.load(f)
        cfg = f"{args.seed}{args.datamodule}{args.disagreement_set}{args.disagreement_proportion}{args.disagreement_from_early_stop_epochs}"
        base_model_cfg = f"{args.seed}{args.datamodule}"
        if args.balanced_sampler:
            cfg += "balanced"
            base_model_cfg += "balanced"

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
    if args.disagreement_set == "train":
        CLASS_WEIGHTS.extend([
            [1.,2.], [1.,3.], [1.,10.], [1.,100.],
            [2.,1.], [3.,1.], [10.,1.], [100.,1.],
        ])
    GAMMAS = [0, 0.5, 1, 2, 4]
    DROPOUTS = [0.1, 0.3, 0.5, 0.7, 0.9]

    if args.datamodule == "waterbirds":
        dm = WaterbirdsDisagreement
    elif args.datamodule == "celeba":
        dm = CelebADisagreement
    args.num_classes = 2
    args.check_val_every_n_epoch = int(args.max_epochs / 10)
    args.ckpt_every_n_epochs = int(args.max_epochs / 10)

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
    args.balanced_sampler = True

    # for testing
    #val_metrics, test_metrics = disagreement(args, gamma=2, dropout=0.5)
    #print(test_metrics)
    #return

    # load current hyperparam search cfg if needed
    full_set_best_worst_group_val = 0
    misclassification_best_worst_group_val = 0
    dropout_best_worst_group_val = 0
    if resume and "full_set_best_worst_group_val" in resume and "full_set_params" in resume and "full_set_metrics" in resume:
        full_set_best_worst_group_val = resume["full_set_best_worst_group_val"]
        full_set_params = resume["full_set_params"]
        full_set_metrics = resume["full_set_metrics"]
    if resume and "misclassification_best_worst_group_val" in resume and "misclassification_params" in resume and "misclassification_metrics" in resume:
        misclassification_best_worst_group_val = resume["misclassification_best_worst_group_val"]
        misclassification_params = resume["misclassification_params"]
        misclassification_metrics = resume["misclassification_metrics"]
    if resume and "dropout_best_worst_group_val" in resume and "dropout_params" in resume and "dropout_metrics" in resume:
        dropout_best_worst_group_val = resume["dropout_best_worst_group_val"]
        dropout_params = resume["dropout_params"]
        dropout_metrics = resume["dropout_metrics"]

    # load current location in hyperparam search cfg
    # just outer loop for now
    start_class_weight_idx = 0
    if resume and "start_class_weight_idx" in resume:
        start_class_weight_idx = resume["start_class_weight_idx"]

    # Do hyperparameter search based on worst group validation error
    for j, class_weights in enumerate(CLASS_WEIGHTS):

        # skip to the right point if resuming
        if j < start_class_weight_idx:
            continue

        # save outer loop location for resuming
        with open("disagreement.pkl", "rb") as f:
            save_state = pickle.load(f)
        save_state[cfg]["start_class_weight_idx"] = j
        with open("disagreement.pkl", "wb") as f:
            pickle.dump(save_state, f)

        print(f"Balanced Full Set DFR: Class Weights {class_weights}")
        val_metrics, test_metrics = disagreement(args, orig_dfr=True, class_weights=class_weights)

        best_worst_group_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
        if best_worst_group_val > full_set_best_worst_group_val:
            full_set_best_worst_group_val = best_worst_group_val
            full_set_params = [class_weights]
            full_set_metrics = [val_metrics, test_metrics]

            with open("disagreement.pkl", "rb") as f:
                save_state = pickle.load(f)
            save_state[cfg]["full_set_best_worst_group_val"] = full_set_best_worst_group_val
            save_state[cfg]["full_set_params"] = full_set_params
            save_state[cfg]["full_set_metrics"] = full_set_metrics
            with open("disagreement.pkl", "wb") as f:
                pickle.dump(save_state, f)

        for gamma in GAMMAS:
            print(f"Balanced Misclassification DFR: Class Weights {class_weights} Gamma {gamma}")
            val_metrics, test_metrics = disagreement(args, gamma=gamma, misclassification_dfr=True, class_weights=class_weights)

            best_worst_group_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
            if best_worst_group_val > misclassification_best_worst_group_val:
                misclassification_best_worst_group_val = best_worst_group_val
                misclassification_params = [class_weights, gamma]
                misclassification_metrics = [val_metrics, test_metrics]

                with open("disagreement.pkl", "rb") as f:
                    save_state = pickle.load(f)
                save_state[cfg]["misclassification_best_worst_group_val"] = misclassification_best_worst_group_val
                save_state[cfg]["misclassification_params"] = misclassification_params
                save_state[cfg]["misclassification_metrics"] = misclassification_metrics
                with open("disagreement.pkl", "wb") as f:
                    pickle.dump(save_state, f)

            for dropout in DROPOUTS:
                print(f"Balanced Dropout Disagreement DFR: Class Weights {class_weights} Gamma {gamma} Dropout {dropout}")
                val_metrics, test_metrics = disagreement(args, gamma=gamma, dropout=dropout, class_weights=class_weights)

                best_worst_group_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
                if best_worst_group_val > dropout_best_worst_group_val:
                    dropout_best_worst_group_val = best_worst_group_val
                    dropout_params = [class_weights, gamma, dropout]
                    dropout_metrics = [val_metrics, test_metrics]

                    with open("disagreement.pkl", "rb") as f:
                        save_state = pickle.load(f)
                    save_state[cfg]["dropout_best_worst_group_val"] = dropout_best_worst_group_val
                    save_state[cfg]["dropout_params"] = dropout_params
                    save_state[cfg]["dropout_metrics"] = dropout_metrics
                    with open("disagreement.pkl", "wb") as f:
                        pickle.dump(save_state, f)

    # save outer loop location for resuming
    with open("disagreement.pkl", "rb") as f:
        save_state = pickle.load(f)
    save_state[cfg]["start_class_weight_idx"] = len(CLASS_WEIGHTS)
    with open("disagreement.pkl", "wb") as f:
        pickle.dump(save_state, f)

    class_weights, gamma, dropout = dropout_params
    print("Rebalancing Ablation")
    _, rebalancing_ablation_metrics = disagreement(args, gamma=gamma, dropout=dropout, class_weights=class_weights, rebalancing=False)
    print("Gamma Ablation")
    _, gamma_ablation_metrics = disagreement(args, gamma=0, dropout=dropout, class_weights=class_weights)
    print("Dropout Ablation")
    _, dropout_ablation_metrics = disagreement(args, gamma=gamma, dropout=dropout, class_weights=class_weights, disagreement_ablation=True)

    print("\n---Hyperparameter Search Results---")
    print("\nERM:")
    print(erm_metrics)
    print("\nFull Set DFR:")
    print(full_set_params)
    print(full_set_metrics)
    print("\nMisclassification DFR:")
    print(misclassification_params)
    print(misclassification_metrics)
    print("\nDropout Disagreement DFR:")
    print(dropout_params)
    print(dropout_metrics)
    print("\nRebalancing Ablation w/ Best Dropout Config")
    print(rebalancing_ablation_metrics)
    print("\nGamma Ablation w/ Best Dropout Config")
    print(gamma_ablation_metrics)
    print("\nDropout Ablation w/ Best Dropout Config")
    print(dropout_ablation_metrics)

if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add("--disagreement_set", choices=("train", "val"))
    parser.add("--disagreement_proportion", type=float)
    parser.add("--disagreement_from_early_stop_epochs", default=0, type=int)

    args = parser.parse_args()

    experiment(args)

