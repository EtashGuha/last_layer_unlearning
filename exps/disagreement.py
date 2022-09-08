from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset

from groundzero.args import parse_args
from groundzero.datamodules.waterbirds import Waterbirds
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.models.resnet import ResNet

TRAIN_TEACHER = False
KL_DIV = True
#TEACHER_WEIGHTS = "lightning_logs/version_168/checkpoints/last.ckpt" # Resnet152 Aux 5epochs


def disagreement(teacher, student, batch_size):
    teacher = teacher.eval()
    student = student.eval()

    dataloader = Waterbirds.val_dataloader()
    indices = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, _ = batch

            logits = student(inputs)
            teacher_logits = teacher(inputs)

            preds = torch.sigmoid(logits) > 0.5
            teacher_preds = torch.sigmoid(teacher_logits) > 0.5

            disagreements = torch.logical_xor(preds, teacher_preds)
            inds = [j + (i * batch_size) for j, d in disagreements if d == True]
            indices.extend(inds)

    new_dataset = Subset(dataloader.dataset, indices)

    # return datamodule with new_dataset as training?

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
    if TRAIN_TEACHER:
        if args.model == "resnet":
            arch = ResNet
        elif args.model == "cnn":
            arch = CNN
    else:
        if args.model == "resnet":
            args.resnet_version = 18
            arch = StudentResNet
        elif args.model == "cnn":
            arch = StudentCNN

    main(args, arch, Waterbirds)


if __name__ == "__main__":
    args = parse_args()
    experiment(args)

