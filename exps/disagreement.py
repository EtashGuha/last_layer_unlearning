"""Main file for Dropout DFR experimentation."""

# Imports Python builtins.
from configargparse import Parser
from copy import deepcopy
from glob import glob
import os
import os.path as osp
import pickle

# Imports groundzero packages.
from groundzero.args import parse_args
from groundzero.datamodules.celeba import CelebADisagreement
from groundzero.datamodules.civilcomments import CivilCommentsDisagreement
from groundzero.datamodules.fmow import FMOWDisagreement
from groundzero.datamodules.multinli import MultiNLIDisagreement
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.imports import valid_models_and_datamodules
from groundzero.main import load_model, main


def get_weights(version, ind=-1):
    list_of_weights = glob(osp.join(os.getcwd(), f"lightning_logs/version_{version}/checkpoints/*"))
    list_of_weights = sorted([w for w in list_of_weights if "best" not in w])
    return list_of_weights[ind]

def get_datamodule_parameters(datamodule):
    if datamodule == "waterbirds":
        datamodule_class = WaterbirdsDisagreement
        num_classes = 2
        train_dist_proportion = [0.7295, 0.0384, 0.2204, 0.0117]
    elif datamodule == "celeba":
        datamodule_class = CelebADisagreement
        num_classes = 2
        train_dist_proportion = [0.4295, 0.4166, 0.1447, 0.0092]
    elif datamodule == "fmow":
        datamodule_class = FMOWDisagreement
        num_classes = 62
        train_dist_proportion = [0.2318, 0.4532, 0.0206, 0.2730, 0.0214]
    elif datamodule == "civilcomments":
        datamodule_class = CivilCommentsDisagreement
        num_classes = 2
        train_dist_proportion = [0.5508, 0.3358, 0.0473, 0.0661]
    elif datamodule == "multinli":
        datamodule_class = MultiNLIDisagreement
        num_classes = 3
        train_dist_proportion = [0.2789, 0.0541, 0.3267, 0.0074, 0.3232, 0.0097]
    else:
        raise ValueError("DataModule not supported.")

    return datamodule_class, num_classes, train_dist_proportion

def load_state():
    if not osp.isfile("disagreement.pkl"):
        # TODO: Initialize the results dict.
        raise NotImplementedError()

    with open("disagreement.pkl", "rb") as f:
        state = pickle.load(f)

    return state

def dump_state(state):
    with open("disagreement.pkl", "wb") as f:
        pickle.dump(state, f)

def reset_fc_hook(model):
    try:
        for layer in model.model.fc:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    except:
        model.model.fc.reset_parameters()

def print_metrics(test_metrics, train_dist_proportion):
    #train_dist_mean_acc = sum([p * group[f"test_acc1/dataloader_idx_{j+1}"] for j, (group, p) in enumerate(zip(test_metrics[1:], train_dist_proportion))])
    #worst_group_acc = min([group[f"test_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(test_metrics[1:])])
    worst_group_acc = min([group[f"test_acc1/dataloader_idx_{j}"] for j, group in enumerate(test_metrics)]) # could be wrong for erm because we do full dataloaders on erm but skip the overall dataloader for dfr for speed

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
    model_class,
    datamodule_class,
    dfr_type,
    num_data,
    dfr_lr,
    dfr_epochs,
    class_balancing=True,
    class_weights=[],
    dropout_prob=0,
    earlystop_weights=None,
    gamma=1,
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

    model = load_model(disagreement_args, model_class)
    earlystop_model = None
    if earlystop_weights:
        early_args = deepcopy(disagreement_args)
        early_args.weights = earlystop_weights
        earlystop_model = load_model(early_args, model_class)

    def disagreement_class(orig_datamodule_class):
        class Disagreement(orig_datamodule_class):
            def __init__(self, args):
                super().__init__(
                    disagreement_args,
                    model=model,
                    earlystop_model=earlystop_model,
                    num_data=num_data,
                    gamma=gamma,
                    class_balancing=class_balancing,
                )

        return Disagreement

    model_hooks = [reset_fc_hook] if reset_fc else None
    _, val_metrics, test_metrics = main(finetune_args, model_class, disagreement_class(datamodule_class), model_hooks=model_hooks)

    return val_metrics, test_metrics

def experiment(args, model_class):
    # Loads save state from pickle file.
    state = load_state()
    curr_state = state[args.datamodule][args.seed]

    # Sets global parameters.
    ERM_CLASS_BALANCING = False
    ERM_CLASS_WEIGHTS = ()
    CLASS_BALANCING = True
    COMBINE_VAL_SET_FOR_ERM = False

    CLASS_WEIGHTS = ()
    #NUM_DATAS = [10, 20, 50, 100, 200]
    NUM_DATAS = [200]

    # Sets search parameters.
    RESET_FC = [False]
    #FULL_DFR_EPOCHS = 100 # Vision
    #DFR_EPOCH_NUMS = [5000] # Vision
    #DFR_LRS = [1e-5, 1e-4, 1e-3, 1e-2] # Vision
    FULL_DFR_EPOCHS = 10 # NLP
    DFR_EPOCH_NUMS = [5000] # NLP
    DFR_LRS = [1e-5] # NLP
    GAMMA = [1]
    EARLYSTOP_NUMS = [1, 2, 5]
    DROPOUT_PROBS = [0.5, 0.7, 0.9]
    
    # Sets datamodule parameters.
    datamodule_class, num_classes, train_dist_proportion = get_datamodule_parameters(args.datamodule)
    args.num_classes = num_classes

    erm_state = curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["erm"][COMBINE_VAL_SET_FOR_ERM]

    # Trains ERM model.
    erm_version = erm_state["version"]
    erm_metrics = erm_state["metrics"]
    if erm_version == -1: #or erm_metrics == []:
        args.balanced_sampler = ERM_CLASS_BALANCING
        args.combine_val_set = COMBINE_VAL_SET_FOR_ERM
        model, erm_val_metrics, erm_test_metrics = main(args, model_class, datamodule_class)
        args.combine_val_set = False

        erm_version = model.trainer.logger.version
        erm_metrics = [erm_val_metrics, erm_test_metrics]
        curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["erm"][COMBINE_VAL_SET_FOR_ERM]["version"] = model.trainer.logger.version
        curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["erm"][COMBINE_VAL_SET_FOR_ERM]["metrics"] = [erm_val_metrics, erm_test_metrics]

        dump_state(state)
        del model
 
    # Gets last-epoch ERM weights.
    args.weights = get_weights(erm_version, ind=-1)

    def dfr_helper(
        dfr_type,
        num_data,
        dfr_lr,
        dropout_prob=0,
        earlystop_num=None,
        gamma=None,
        dfr_epoch_num=None,
        dfr_epochs=None,
        reset_fc=False,
    ):
        earlystop_weights = None
        if earlystop_num:
            earlystop_weights = get_weights(erm_version, ind=earlystop_num-1)

        if dfr_epoch_num:
            dfr_epochs = dfr_epoch_num // num_data
        elif not dfr_epochs:
            raise ValueError("Must specify either dfr_epochs or dfr_epoch_num")

        val_metrics, test_metrics = dfr(
            args,
            model_class,
            datamodule_class,
            dfr_type,
            num_data,
            dfr_lr,
            dfr_epochs,
            class_balancing=CLASS_BALANCING,
            class_weights=CLASS_WEIGHTS,
            dropout_prob=dropout_prob,
            earlystop_weights=earlystop_weights,
            gamma=gamma,
            reset_fc=reset_fc,
        )
                                
        if num_data == "all":
            orig = True if dfr_type == "orig" else False
            val = curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["dfn"][CLASS_BALANCING][CLASS_WEIGHTS]["dfr"][orig]["val"]
        else:
            val = curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["dfn"][CLASS_BALANCING][CLASS_WEIGHTS][dfr_type][num_data]["val"]

        #worst_wg_val = min([group[f"val_acc1/dataloader_idx_{j+1}"] for j, group in enumerate(val_metrics[1:])])
        worst_wg_val = min([group[f"val_acc1/dataloader_idx_{j}"] for j, group in enumerate(val_metrics)])
        if worst_wg_val >= val:
            if dfr_epoch_num:
                params = [dfr_epoch_num, dfr_lr]
            else:
                params = [dfr_epochs, dfr_lr]

            if gamma:
                params.append(gamma)
            if earlystop_weights:
                params.append(earlystop_num)
            if dropout_prob:
                params.append(dropout_prob)

            if num_data == "all":
                orig = True if dfr_type == "orig" else False
                curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["dfn"][CLASS_BALANCING][CLASS_WEIGHTS]["dfr"][orig]["val"] = worst_wg_val
                curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["dfn"][CLASS_BALANCING][CLASS_WEIGHTS]["dfr"][orig]["params"] = params
                curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["dfn"][CLASS_BALANCING][CLASS_WEIGHTS]["dfr"][orig]["metrics"] = [val_metrics, test_metrics]
            else:
                curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["dfn"][CLASS_BALANCING][CLASS_WEIGHTS][dfr_type][num_data]["val"] = worst_wg_val
                curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["dfn"][CLASS_BALANCING][CLASS_WEIGHTS][dfr_type][num_data]["params"] = params
                curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["dfn"][CLASS_BALANCING][CLASS_WEIGHTS][dfr_type][num_data]["metrics"] = [val_metrics, test_metrics]

    #print(f"Group-Balanced Full DFR")
    #dfr_helper("orig", "all", dfr_epochs=FULL_DFR_EPOCHS, dfr_lr=args.lr, reset_fc=True)

    #print(f"Group-Unbalanced Full DFR")
    #dfr_helper("random", "all", dfr_epochs=FULL_DFR_EPOCHS, dfr_lr=args.lr, reset_fc=True)
    #return

    # Does hyperparameter search based on worst group validation error.
    for dfr_lr in DFR_LRS:
        for reset_fc in RESET_FC:
            for num_data in NUM_DATAS:
                for dfr_epoch_num in DFR_EPOCH_NUMS:
                    def dfr_helper2(dfr_type, gamma=None, dropout_prob=0, earlystop_num=None):
                        return dfr_helper(dfr_type, num_data, dfr_lr, dfr_epoch_num=dfr_epoch_num,
                                reset_fc=reset_fc, gamma=gamma, dropout_prob=dropout_prob, earlystop_num=earlystop_num)

                    print(f"Random DFN {num_data}: Epoch Num {dfr_epoch_num} LR {dfr_lr}")
                    dfr_helper2("random")

                    for gamma in GAMMA:
                        print(f"Misclassification DFN {num_data}: Gamma {gamma} Epoch Num {dfr_epoch_num} LR {dfr_lr}")
                        dfr_helper2("miscls", gamma=gamma)

                        for dropout_prob in DROPOUT_PROBS:
                            print(f"Dropout DFN {num_data}: Gamma {gamma} Epoch Num {dfr_epoch_num} Dropout {dropout_prob} LR {dfr_lr}")
                            dfr_helper2("dropout", gamma=gamma, dropout_prob=dropout_prob)

                        for earlystop_num in EARLYSTOP_NUMS:
                            print(f"Early Stop Disagreement DFN {num_data}: Gamma {gamma} Epoch Num {dfr_epoch_num} Early Stop {earlystop_num} LR {dfr_lr}")
                            dfr_helper2("earlystop", gamma=gamma, earlystop_num=earlystop_num)

                            print(f"Early Stop Misclassification DFN {num_data}: Gamma {gamma} Epoch Num {dfr_epoch_num} Early Stop {earlystop_num} LR {dfr_lr}")
                            dfr_helper2("earlystop_miscls", gamma=gamma, earlystop_num=earlystop_num)

    disk_state = load_state()
    state.update(disk_state)
    dump_state(state)

    dfn_state = curr_state[ERM_CLASS_BALANCING][ERM_CLASS_WEIGHTS]["dfn"][CLASS_BALANCING][CLASS_WEIGHTS]

    print("\n---Hyperparameter Search Results---")
    """
    print("\nERM:")
    print_metrics(erm_metrics[1], train_dist_proportion)
    print("\nGroup-Balanced Full DFR:")
    print(dfn_state["dfr"][True]["params"])
    print_metrics(dfn_state["dfr"][True]["metrics"][1], train_dist_proportion)
    print("\nGroup-Unbalanced Full DFR:")
    print(dfn_state["dfr"][False]["params"])
    print_metrics(dfn_state["dfr"][False]["metrics"][1], train_dist_proportion)
    """

    for num_data in NUM_DATAS:
        #print(f"\nGroup-Balanced DFN {num_data}:")
        #print(orig[num][CLASS_BALANCING]["params"])
        #print_metrics(orig[num][CLASS_BALANCING]["metrics"][1], train_dist_proportion)
        print(f"\nRandom DFN {num_data}:")
        print(dfn_state["random"][num_data]["params"])
        print_metrics(dfn_state["random"][num_data]["metrics"][1], train_dist_proportion)
        print(f"\nMisclassification DFN {num_data}:")
        print(dfn_state["miscls"][num_data]["params"])
        print_metrics(dfn_state["miscls"][num_data]["metrics"][1], train_dist_proportion)
        print(f"\nDropout DFN {num_data}:")
        print(dfn_state["dropout"][num_data]["params"])
        print_metrics(dfn_state["dropout"][num_data]["metrics"][1], train_dist_proportion)
        print(f"\nEarly Stop Disagreement DFN {num_data}:")
        print(dfn_state["earlystop"][num_data]["params"])
        print_metrics(dfn_state["earlystop"][num_data]["metrics"][1], train_dist_proportion)
        print(f"\nEarly Stop Misclassification DFN {num_data}:")
        print(dfn_state["earlystop_miscls"][num_data]["params"])
        print_metrics(dfn_state["earlystop_miscls"][num_data]["metrics"][1], train_dist_proportion)

    
if __name__ == "__main__":
    args = parse_args()

    models, _ = valid_models_and_datamodules()
        
    experiment(args, models[args.model])

