from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset

from groundzero.args import parse_args
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.main import load_model, main
from groundzero.models.resnet import ResNet

MODE = "disagreement_teacher" # "train_teacher", "distillation", "disagreement_student", "disagreement_teacher"
GAMMA = 1
DISAGREEMENT_SET = "val"
DISAGREEMENT_PROPORTION = 0.5
DISAGREEMENT_LOSS = "bce" # "kl", "bce"
MISCLASSIFICATION_DFR = False # for misclassification error disagreement, JTT-style
FULL_SET_DFR = False # for full val set DFR
REBALANCING = True
DROPOUT = True

TEACHER_WEIGHTS = "lightning_logs/version_246/checkpoints/last.ckpt" # ERM ResNet50 p=0.8 val
STUDENT_WEIGHTS = "lightning_logs/version_246/checkpoints/last.ckpt" # ERM ResNet50 p=0.8 val
# TEACHER_WEIGHTS = "lightning_logs/version_260/checkpoints/last.ckpt" # ResNet50 p=0.8 val misclassification DFR from 246
# TEACHER_WEIGHTS = "lightning_logs/version_258/checkpoints/last.ckpt" # ResNet50 p=0.8 val disagreement DFR from 248
# STUDENT_WEIGHTS = "lightning_logs/version_247/checkpoints/last.ckpt" # ERM ResNet18 p=0.8 val
# STUDENT_WEIGHTS = "lightning_logs/version_248/checkpoints/last.ckpt" # ResNet18 p=0.8 val distillation from 246
# STUDENT_WEIGHTS = "lightning_logs/version_262/checkpoints/last.ckpt" # ResNet18 p=0.8 val distillation from 258

class Waterbirds2(WaterbirdsDisagreement):
    def __init__(self, args, student=None, gamma=None, misclassification_dfr=None, full_set_dfr=None, dropout=False):
        super().__init__(
            args,
            disagreement_set=DISAGREEMENT_SET,
            disagreement_proportion=DISAGREEMENT_PROPORTION,
            student=student,
            gamma=gamma,
            misclassification_dfr=misclassification_dfr,
            full_set_dfr=full_set_dfr,
            dropout=dropout,
            rebalancing=REBALANCING,
        )

def new_logits(logits):
    """Sigmoid to probability distribution for softmax."""
    col1 = logits.unsqueeze(1)
    col2 = -logits.unsqueeze(1)

    return torch.cat((col1, col2), 1)

def step_helper(self, batch):
    inputs, targets = batch

    logits = self(inputs)
    teacher_logits = self.teacher(inputs)
    
    if self.hparams.num_classes == 1:
        probs = torch.sigmoid(logits).detach().cpu()
        loss = F.kl_div(
            F.logsigmoid(new_logits(logits)),
            torch.sigmoid(new_logits(teacher_logits)),
            reduction="batchmean",
        )
    else:
        probs = F.softmax(logits, dim=1).detach().cpu()
        teacher_probs = F.softmax(teacher_logits, dim=1)
        loss = F.kl_div(
            F.log_softmax(logits, dim=1),
            teacher_probs,
            reduction="batchmean",
        )
     
    targets = targets.cpu()

    return {"loss": loss, "probs": probs, "targets": targets}

class StudentResNet(ResNet):
    def __init__(self, args):
        super().__init__(args)

        teacher_args = deepcopy(args)
        teacher_args.resnet_version = 50
        self.teacher = ResNet(teacher_args)
        state_dict = torch.load(TEACHER_WEIGHTS, map_location="cpu")["state_dict"]
        self.teacher.load_state_dict(state_dict, strict=False)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def step(self, batch, idx):
        if (MODE == "distillation") or (MODE == "disagreement" and DISAGREEMENT_LOSS == "kl"):
            return step_helper(self, batch)
        else:
            return super().step(batch, idx)

def experiment(args):
    if MODE == "train_teacher":
        main(args, ResNet, Waterbirds2)
    elif MODE == "distillation":
        args.resnet_version = 18
        main(args, StudentResNet, Waterbirds2)
    elif MODE in ("disagreement_student", "disagreement_teacher"):
        args.train_fc_only = True
        args.lr = 0.01
        args.max_epochs = 100
        args.check_val_every_n_epoch = 100
        #args.check_val_every_n_epoch = 50
        args.num_classes = 1

        #args.resnet_version = 18
        #arch = StudentResNet
        arch = ResNet
        if DROPOUT:
            args.dropout_prob = 0.5
        
        args.weights = STUDENT_WEIGHTS
        model = load_model(args, arch)
        if DROPOUT:
            args.dropout_prob = 0

        class Waterbirds3(Waterbirds2):
            def __init__(self, args):
                super().__init__(
                    args,
                    student=model,
                    gamma=GAMMA,
                    misclassification_dfr=MISCLASSIFICATION_DFR,
                    full_set_dfr=FULL_SET_DFR,
                    dropout=DROPOUT,
                )

        if MODE == "disagreement_teacher":
            args.weights = TEACHER_WEIGHTS
            args.resnet_version = 50
            arch = ResNet

        main(args, arch, Waterbirds3)

def disagreement(args, gamma, misclassification_dfr=False, full_set_dfr=False):
    args.train_fc_only = True
    args.max_epochs = 100
    args.check_val_every_n_epoch = 100
    args.num_classes = 1
    
    model = load_model(args, ResNet)

    class Waterbirds3(Waterbirds2):
        def __init__(self, args):
            super().__init__(
                args,
                student=model,
                gamma=gamma,
                misclassification_dfr=misclassification_dfr,
                full_set_dfr=full_set_dfr,
                dropout=(args.dropout_prob != 0),
            )

    main(args, ResNet, Waterbirds3)

def run_all_dropout(args):
    args.check_val_every_n_epoch = 50
    #seeds = (42, 43, 44)
    lambdas = (1e-4, 1e-3, 1e-2)
    gammas = (0.5, 1, 2)
    dropouts = (0.25, 0.5, 0.75)
    dfr_lrs = (0.1, 0.03, 0.01)

    args.weight_decay = 1e-4
    args.l1_regularization = 0
    args.dropout_prob = 0
    model, _ = main(args, ResNet, Waterbirds2)

    version = model.trainer.logger.version
    args.weights = f"lightning_logs/version_{version}/checkpoints/last.ckpt"
        
    for lr in dfr_lrs:
        args.lr = lr
        for lmbda in lambdas:
            args.weight_decay = 0
            args.l1_regularization = lmbda
            print(f"Full Set DFR Lambda {lmbda} LR {lr}")
            disagreement(args, 1, full_set_dfr=True)

            for gamma in gammas:
                print(f"Misclassification DFR Lambda {lmbda} Gamma {gamma} LR {lr}")
                disagreement(args, gamma, misclassification_dfr=True)

                for dropout in dropouts:
                    args.dropout_prob = dropout
                    print(f"Dropout Disagreement DFR Lambda {lmbda} Gamma {gamma} Dropout {dropout} LR {lr}")
                    disagreement(args, gamma)

                    print(f"Dropout Misclassification DFR Lambda {lmbda} Gamma {gamma} Dropout {dropout} LR {lr}")
                    disagreement(args, gamma, misclassification_dfr=True)

if __name__ == "__main__":
    args = parse_args()
    #experiment(args)
    run_all_dropout(args)

