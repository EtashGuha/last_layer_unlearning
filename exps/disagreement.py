from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset

from groundzero.args import parse_args
from groundzero.datamodules.waterbirds import WaterbirdsDisagreement
from groundzero.main import load_model, main
from groundzero.models.cnn import CNN
from groundzero.models.resnet import ResNet

MODE = "distillation" # "train_teacher", "distillation", "disagreement"
LAMBDA = 0.5
DISAGREEMENT_SET = "val"
DISAGREEMENT_PROPORTION = 0.8
TEACHER_WEIGHTS = "lightning_logs/version_186/checkpoints/last.ckpt" # Resnet152 5epochs

class Waterbirds2(WaterbirdsDisagreement):
    def __init__(self, args, student=None, lmbda=None):
        super().__init__(args, disagreement_set=DISAGREEMENT_SET, disagreement_proportion=DISAGREEMENT_PROPORTION, student=student, lmbda=lmbda)

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
        teacher_args.resnet_version = 152
        self.teacher = ResNet(teacher_args)
        state_dict = torch.load(TEACHER_WEIGHTS, map_location="cpu")["state_dict"]
        self.teacher.load_state_dict(state_dict, strict=False)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def step(self, batch, idx):
        return step_helper(self, batch)

class StudentCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
        
        teacher_args = deepcopy(args)
        teacher_args.resnet_version = 152
        self.teacher = ResNet(teacher_args)
        state_dict = torch.load(TEACHER_WEIGHTS, map_location="cpu")["state_dict"]
        self.teacher.load_state_dict(state_dict, strict=False)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def step(self, batch, idx):
        return step_helper(self, batch)

def experiment(args):
    if MODE == "train_teacher":
        if args.model == "resnet":
            arch = ResNet
        elif args.model == "cnn":
            arch = CNN
        main(args, arch, Waterbirds2)
    elif MODE == "distillation":
        if args.model == "resnet":
            args.resnet_version = 18
            arch = StudentResNet
        elif args.model == "cnn":
            arch = StudentCNN
        main(args, arch, Waterbirds2)
    elif MODE == "disagreement":
        if not args.weights:
            raise ValueError()

        args.train_fc_only = True
        args.num_classes = 1

        if args.model == "resnet":
            args.resnet_version = 18
            arch = StudentResNet
        elif args.model == "cnn":
            arch = StudentCNN

        model = load_model(args, arch)

        class Waterbirds3(Waterbirds2):
            def __init__(self, args):
                super().__init__(args, student=model, lmbda=LAMBDA)

        main(args, arch, Waterbirds3)

if __name__ == "__main__":
    args = parse_args()
    experiment(args)

