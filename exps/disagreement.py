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
from groundzero.datamodules.multinli import MultiNLIDisagreement
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.main import load_model, main
from groundzero.models.bert import BERT
from groundzero.models.resnet import ResNet


def get_weights(version, ind=-1):
    list_of_weights = glob(osp.join(os.getcwd(), f"lightning_logs/version_{version}/checkpoints/*"))
    list_of_weights = sorted([w for w in list_of_weights if "best" not in w])
    return list_of_weights[ind]

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

def reset_fc_hook(model):
    try:
        for layer in model.model.fc:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    except:
        model.model.fc.reset_parameters()

def save_metrics(cfg, metrics, name=None):
    with open("disagreement.pkl", "rb") as f:
        save_state = pickle.load(f)
        if name:
            save_state[cfg][name] = metrics
        else:
            save_state[cfg] = metrics
    with open("disagreement.pkl", "wb") as f:
        pickle.dump(save_state, f)

def print_metrics(test_metrics, train_dist_proportion):
    #train_dist_mean_acc = sum([p * group[f"test_acc1/dataloader_idx_{j+1}"] for j, (group, p) in enumerate(zip(test_metrics[1:], train_dist_proportion))])
    #worst_group_acc = min([group[f"test_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(test_metrics[1:])])
    worst_group_acc = min([group[f"test_acc1/dataloader_idx_{j}"] for j, group in enumerate(test_metrics)])

    """
    test_dist_mean_acc = test_metrics[0]["test_acc1/dataloader_idx_0"]
    train_dist_mean_acc = round(train_dist_mean_acc * 100, 1)
    test_dist_mean_acc = round(test_dist_mean_acc * 100, 1)
    """
    worst_group_acc = round(worst_group_acc * 100, 1)

    """
    print(f"Train Dist Mean Acc: {train_dist_mean_acc}")
    print(f"Test Dist Mean Acc: {test_dist_mean_acc}")
    """
    print(f"Worst Group Acc: {worst_group_acc}")

def dfr(
    args,
    dfr_type,
    class_labels_num,
    class_balancing=True,
    class_weights=[],
    dfr_epochs=100,
    dropout_prob=None,
    early_stop_weights=None,
    gamma=None,
    dfr_lr=None,
    reset_fc=False,
):
    disagreement_args = deepcopy(args)
    disagreement_args.dfr_type = dfr_type
    disagreement_args.dropout_prob = dropout_prob

    finetune_args = deepcopy(args)
    finetune_args.train_fc_only = True
    finetune_args.check_val_every_n_epoch = dfr_epochs + 1 # change to dfr_epochs to save model
    finetune_args.ckpt_every_n_epochs = dfr_epochs + 1 # change to dfr_epochs to save model
    finetune_args.max_epochs = dfr_epochs
    finetune_args.class_weights = class_weights
    finetune_args.lr = dfr_lr
    finetune_args.lr_scheduler = "step"
    finetune_args.lr_steps = []

    early_stop_model = None
    if early_stop_weights:
        early_args = deepcopy(disagreement_args)
        early_args.weights = early_stop_weights

    def disagreement_cls(orig_cls):
        class Disagreement(orig_cls):
            def __init__(self, args):
                super().__init__(
                    disagreement_args,
                    model=model,
                    early_stop_model=early_stop_model,
                    class_balancing=class_balancing,
                    class_labels_num=class_labels_num,
                    gamma=gamma,
                )

        return Disagreement

    model_hooks = []
    if reset_fc:
        model_hooks = [reset_fc_hook]

    if args.datamodule == "waterbirds":
        model = load_model(disagreement_args, ResNet)
        if early_stop_weights:
            early_stop_model = load_model(early_args, ResNet)
        _, val_metrics, test_metrics = main(finetune_args, ResNet, disagreement_cls(WaterbirdsDisagreement), model_hooks=model_hooks)
    elif args.datamodule == "celeba":
        model = load_model(disagreement_args, ResNet)
        if early_stop_weights:
            early_stop_model = load_model(early_args, ResNet)
        _, val_metrics, test_metrics = main(finetune_args, ResNet, disagreement_cls(CelebADisagreement), model_hooks=model_hooks)
    elif args.datamodule == "fmow":
        model = load_model(disagreement_args, ResNet)
        if early_stop_weights:
            early_stop_model = load_model(early_args, ResNet)
        _, val_metrics, test_metrics = main(finetune_args, ResNet, disagreement_cls(FMOWDisagreement), model_hooks=model_hooks)
    elif args.datamodule == "civilcomments":
        model = load_model(disagreement_args, BERT)
        if early_stop_weights:
            early_stop_model = load_model(early_args, BERT)
        _, val_metrics, test_metrics = main(finetune_args, BERT, disagreement_cls(CivilCommentsDisagreement), model_hooks=model_hooks)
    elif args.datamodule == "multinli":
        model = load_model(disagreement_args, BERT)
        if early_stop_weights:
            early_stop_model = load_model(early_args, BERT)
        _, val_metrics, test_metrics = main(finetune_args, BERT, disagreement_cls(MultiNLIDisagreement), model_hooks=model_hooks)
    else:
        raise ValueError("DataModule not supported.")

    return val_metrics, test_metrics

def experiment(args):
    # Loads save state from pickle file.
    cfg, erm_cfg, resume, erm_resume = load_save_state(args)

    # Sets search parameters.
    NUMS = [10, 20, 50, 100, 200]
    #DFR_EPOCH_NUMS = [500, 1000, 2000, 5000]
    DFR_EPOCH_NUMS = [5000]
    #GAMMA = [0.5, 1]
    GAMMA = [1]
    EARLY_STOP_INDS = [1, 2, 5]
    DROPOUT_PROBS = [0.5, 0.7, 0.9]
    
    # Sets datamodule-specific parameters.
    CLASS_WEIGHTS = []
    CLASS_BALANCING = True
    RESET_FC = False
    if args.datamodule == "waterbirds":
        model_cls = ResNet
        dm = WaterbirdsDisagreement
        args.num_classes = 2
        #RESET_FC = True
        #DFR_LRS = [1e-5, 1e-4, 1e-3, 1e-2]
        DFR_LRS = [1e-3, 1e-2]
        TRAIN_DIST_PROPORTION = [0.7295, 0.0384, 0.2204, 0.0117]
    elif args.datamodule == "celeba":
        model_cls = ResNet
        dm = CelebADisagreement
        args.num_classes = 2
        CLASS_WEIGHTS = [1, 2]
        DFR_LRS = [1e-5, 1e-4, 1e-3, 1e-2]
        TRAIN_DIST_PROPORTION = [0.4295, 0.4166, 0.1447, 0.0092]
    elif args.datamodule == "fmow":
        model_cls = ResNet
        dm = FMOWDisagreement
        args.num_classes = 62
        CLASS_BALANCING = False
        DFR_LRS = [1e-5, 1e-4, 1e-3, 1e-2]
        TRAIN_DIST_PROPORTION = [0.2318, 0.4532, 0.0206, 0.2730, 0.0214]
    elif args.datamodule == "civilcomments":
        model_cls = BERT
        dm = CivilCommentsDisagreement
        args.num_classes = 2
        TRAIN_DIST_PROPORTION = [0.5508, 0.3358, 0.0473, 0.0661]
    elif args.datamodule == "multinli":
        model_cls = BERT
        dm = MultiNLIDisagreement
        args.num_classes = 3
        TRAIN_DIST_PROPORTION = [0.2789, 0.0541, 0.3267, 0.0074, 0.3232, 0.0097]
    else:
        raise ValueError("DataModule not supported.")

    try:
        # Loads ERM model.
        #erm_version = erm_resume[CLASS_BALANCING][CLASS_WEIGHTS]["version"]
        #erm_metrics = erm_resume[CLASS_BALANCING][CLASS_WEIGHTS]["metrics"]
        erm_version = erm_resume["version"]
        erm_metrics = erm_resume["metrics"]
    except:
        """
        # Resumes ERM training if interrupted (need to manually add version).
        if erm_resume:
            erm_version = erm_resume[CLASS_BALANCING]["version"]
            args.weights = get_weights(erm_version, key="latest")
            args.resume_training = True
        """
        if not erm_resume:
            erm_resume = {label: {w: {"version": -1, "metrics": []} for w in [[], [1, 2]]} for label in [True, False]}

        model, erm_val_metrics, erm_test_metrics = main(args, model_cls, dm)
        erm_resume[CLASS_BALANCING][CLASS_WEIGHTS]["version"] = model.trainer.logger.version
        erm_resume[CLASS_BALANCING][CLASS_WEIGHTS]["metrics"] = [erm_val_metrics, erm_test_metrics]
        #args.weights = ""
        #args.resume_training = False
        del model

        save_metrics(erm_cfg, erm_resume)
 
    args.weights = get_weights(erm_version, ind=-1)

    # Loads current hyperparam search cfg if needed.
    orig = {num: {label: {"val": -1} for label in [True, False]} for num in NUMS}
    random = deepcopy(orig)
    miscls = deepcopy(orig)
    dropout = deepcopy(orig)
    earlystop = deepcopy(orig)
    earlystop_miscls = deepcopy(orig)
    orig.update({"all": {label: {"val": -1} for label in [True, False]}})
    random.update({"all": {label: {"val": -1} for label in [True, False]}})
    if resume:
        orig = resume["orig"] if "orig" in resume else orig
        random = resume["random"] if "random" in resume else random
        miscls = resume["miscls"] if "miscls" in resume else miscls
        dropout = resume["dropout"] if "dropout" in resume else dropout
        earlystop = resume["earlystop"] if "earlystop" in resume else earlystop
        earlystop_miscls = resume["earlystop_miscls"] if "earlystop_miscls" in resume else earlystop_miscls

    results = {"orig": orig, "random": random, "miscls": miscls, "dropout": dropout, "earlystop": earlystop, "earlystop_miscls": earlystop_miscls}
    def dfr_helper(
        args,
        dfr_type,
        class_labels_num,
        dropout_prob=0,
        early_stop_ind=None,
        gamma=None,
        dfr_epoch_num=None,
        dfr_epochs=None,
        dfr_lr=None,
        reset_fc=False,
    ):

        if early_stop_ind:
            early_stop_weights = get_weights(erm_version, ind=early_stop_ind-1)
        else:
            early_stop_weights = None

        if dfr_epoch_num:
            dfr_epochs = dfr_epoch_num // class_labels_num

        val_metrics, test_metrics = dfr(
            args,
            dfr_type,
            class_labels_num,
            class_balancing=CLASS_BALANCING,
            class_weights=CLASS_WEIGHTS,
            dfr_epochs=dfr_epochs,
            dropout_prob=dropout_prob,
            early_stop_weights=early_stop_weights,
            gamma=gamma,
            dfr_lr=dfr_lr,
            reset_fc=reset_fc,
        )

        #worst_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
        worst_wg_val = min([group[f"val_acc1/dataloader_idx_{j}"] for j, group in enumerate(val_metrics)])
        if worst_wg_val >= results[dfr_type][class_labels_num][CLASS_BALANCING]["val"]:
            params = [CLASS_WEIGHTS, dfr_epoch_num, dfr_lr]
            if gamma:
                params.append(gamma)
            if early_stop_weights:
                params.append(early_stop_ind)
            if dropout_prob:
                params.append(dropout_prob)

            results[dfr_type][class_labels_num][CLASS_BALANCING]["val"] = worst_wg_val
            results[dfr_type][class_labels_num][CLASS_BALANCING]["params"] = params
            results[dfr_type][class_labels_num][CLASS_BALANCING]["metrics"] = [val_metrics, test_metrics]
            save_metrics(cfg, results[dfr_type], dfr_type)

    # Does hyperparameter search based on worst group validation error.
    for dfr_lr in DFR_LRS:
        if dfr_lr != 1e-3:
            print(f"Group-Balanced Full DFR: LR {dfr_lr}")
            dfr_helper(args, "orig", "all", dfr_epochs=50, dfr_lr=dfr_lr, reset_fc=True)

            print(f"Group-Unbalanced Full DFR: LR {dfr_lr}")
            dfr_helper(args, "random", "all", dfr_epochs=50, dfr_lr=dfr_lr, reset_fc=True)

        for num in NUMS:
            for dfr_epoch_num in DFR_EPOCH_NUMS:
                #print(f"Group-Balanced DFN {num}: Epoch Num {dfr_epoch_num} LR {dfr_lr}")
                #dfr_helper(args, "orig", num, dfr_epoch_num=dfr_epoch_num, dfr_lr=dfr_lr, reset_fc=RESET_FC)

                if not (num == 100 and dfr_lr == 1e-3):
                    print(f"Random DFN {num}: Epoch Num {dfr_epoch_num} LR {dfr_lr}")
                    dfr_helper(args, "random", num, dfr_epoch_num=dfr_epoch_num, dfr_lr=dfr_lr, reset_fc=RESET_FC)

                for gamma in GAMMA:
                    if not (num == 100 and dfr_lr == 1e-3):
                        print(f"Misclassification DFN {num}: Gamma {gamma} Epoch Num {dfr_epoch_num} LR {dfr_lr}")
                        dfr_helper(args, "miscls", num, gamma=gamma, dfr_epoch_num=dfr_epoch_num, dfr_lr=dfr_lr, reset_fc=RESET_FC)

                        for dropout_prob in DROPOUT_PROBS:
                            print(f"Dropout DFN {num}: Gamma {gamma} Epoch Num {dfr_epoch_num} Dropout {dropout_prob} LR {dfr_lr}")
                            dfr_helper(args, "dropout", num, dropout_prob=dropout_prob, gamma=gamma, dfr_epoch_num=dfr_epoch_num, dfr_lr=dfr_lr, reset_fc=RESET_FC)

                    for early_stop_ind in EARLY_STOP_INDS:
                        print(f"Early Stop Disagreement DFN {num}: Gamma {gamma} Epoch Num {dfr_epoch_num} Early Stop {early_stop_ind} LR {dfr_lr}")
                        dfr_helper(args, "earlystop", num, early_stop_ind=early_stop_ind, gamma=gamma, dfr_epoch_num=dfr_epoch_num, dfr_lr=dfr_lr, reset_fc=RESET_FC)

                        print(f"Early Stop Misclassification DFN {num}: Gamma {gamma} Epoch Num {dfr_epoch_num} Early Stop {early_stop_ind} LR {dfr_lr}")
                        dfr_helper(args, "earlystop_miscls", num, early_stop_ind=early_stop_ind, gamma=gamma, dfr_epoch_num=dfr_epoch_num, dfr_lr=dfr_lr, reset_fc=RESET_FC)

    print("\n---Hyperparameter Search Results---")
    print("\nERM:")
    print_metrics(erm_metrics[1], TRAIN_DIST_PROPORTION)
    print("\nGroup-Balanced Full DFR:")
    print(orig["all"][CLASS_BALANCING]["params"])
    print_metrics(orig["all"][CLASS_BALANCING]["metrics"][1], TRAIN_DIST_PROPORTION)
    print("\nGroup-Unbalanced Full DFR:")
    print(random["all"][CLASS_BALANCING]["params"])
    print_metrics(random["all"][CLASS_BALANCING]["metrics"][1], TRAIN_DIST_PROPORTION)

    for num in NUMS:
        #print(f"\nGroup-Balanced DFR {num}:")
        #print(orig[num][CLASS_BALANCING]["params"])
        #print_metrics(orig[num][CLASS_BALANCING]["metrics"][1], TRAIN_DIST_PROPORTION)
        print(f"\nRandom DFR {num}:")
        print(random[num][CLASS_BALANCING]["params"])
        print_metrics(random[num][CLASS_BALANCING]["metrics"][1], TRAIN_DIST_PROPORTION)
        print(f"\nMisclassification DFR {num}:")
        print(miscls[num][CLASS_BALANCING]["params"])
        print_metrics(miscls[num][CLASS_BALANCING]["metrics"][1], TRAIN_DIST_PROPORTION)
        print(f"\nDropout DFR {num}:")
        print(dropout[num][CLASS_BALANCING]["params"])
        print_metrics(dropout[num][CLASS_BALANCING]["metrics"][1], TRAIN_DIST_PROPORTION)
        print(f"\nEarly Stop Disagreement DFR {num}:")
        print(earlystop[num][CLASS_BALANCING]["params"])
        print_metrics(earlystop[num][CLASS_BALANCING]["metrics"][1], TRAIN_DIST_PROPORTION)
        print(f"\nEarly Stop Misclassification DFR {num}:")
        print(earlystop_miscls[num][CLASS_BALANCING]["params"])
        print_metrics(earlystop_miscls[num][CLASS_BALANCING]["metrics"][1], TRAIN_DIST_PROPORTION)

    
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

