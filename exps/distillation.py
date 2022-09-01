from copy import deepcopy

import torch
import torch.nn.functional as F

from groundzero.args import parse_args
from groundzero.datamodules.waterbirds import Waterbirds
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.models.resnet import ResNet

TEACHER_WEIGHTS = "lightning_logs/version_0/checkpoints/epoch=00-val_loss=0.292-val_acc1=0.900.ckpt"
#TEACHER_WEIGHTS = "lightning_logs/version_0/checkpoints/last.ckpt"


"""
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)
"""

def new_logits(logits):
    col1 = logits.unsqueeze(1)
    col2 = -logits.unsqueeze(1)

    return torch.cat((col1, col2), 1)

class DistillationResNet(ResNet):
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
        inputs, targets = batch

        logits = self(inputs)
        teacher_logits = self.teacher(inputs)

        loss = F.mse_loss(logits, teacher_logits)

        if self.hparams.num_classes == 1:
            probs = torch.sigmoid(logits).detach().cpu()
            #loss = F.kl_div(F.logsigmoid(new_logits(logits)), F.sigmoid(new_logits(teacher_logits)), reduction="batchmean")
        else:
            probs = F.softmax(logits, dim=1).detach().cpu()
            #teacher_probs = F.softmax(teacher_logits, dim=1)
            #loss = F.kl_div(F.log_softmax(logits, dim=1), teacher_probs, reduction="batchmean")

        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}

class DistillationCNN(CNN):
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
        inputs, targets = batch

        logits = self(inputs)
        teacher_logits = self.teacher(inputs)

        loss = F.mse_loss(logits, teacher_logits)

        if self.hparams.num_classes == 1:
            probs = torch.sigmoid(logits).detach().cpu()
            #loss = F.kl_div(F.logsigmoid(new_logits(logits)), F.sigmoid(new_logits(teacher_logits)), reduction="batchmean")
        else:
            probs = F.softmax(logits, dim=1).detach().cpu()
            #teacher_probs = F.softmax(teacher_logits, dim=1)
            #loss = F.kl_div(F.log_softmax(logits, dim=1), teacher_probs, reduction="batchmean")

        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}

def experiment(args):
    m = None
    if args.model == "resnet":
        args.resnet_version = 18
        m = DistillationResNet
    elif args.model == "cnn":
        m = DistillationCNN

    main(args, m, Waterbirds)


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
