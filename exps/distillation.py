from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from groundzero.args import parse_args
from groundzero.datamodules.waterbirds import Waterbirds
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.models.resnet import ResNet

TRAIN_TEACHER = True
REG = None # None, "mlp", "aux"
TRAIN_W_REG = False
K = 32
LAMBDA = 0.01
KL_DIV = True
#TEACHER_WEIGHTS = "lightning_logs/version_0/checkpoints/epoch=00-val_loss=0.292-val_acc1=0.900.ckpt"
#TEACHER_WEIGHTS = "lightning_logs/version_0/checkpoints/last.ckpt"
#TEACHER_WEIGHTS = "lightning_logs/version_144/checkpoints/last.ckpt" # Resnet152 MLP 50epochs
#TEACHER_WEIGHTS = "lightning_logs/version_145/checkpoints/last.ckpt" # Resnet152 MLP 5epochs
#TEACHER_WEIGHTS = "lightning_logs/version_171/checkpoints/last.ckpt" # Resnet152 Aux 50epochs
#TEACHER_WEIGHTS = "lightning_logs/version_168/checkpoints/last.ckpt" # Resnet152 Aux 5epochs


def new_logits(logits):
    col1 = logits.unsqueeze(1)
    col2 = -logits.unsqueeze(1)

    return torch.cat((col1, col2), 1)

outputs = {}
def get_output(name):
    def hook(model, input, output):
        outputs[name] = output
    return hook

def step_helper(self, batch):
    inputs, targets = batch

    output = self(inputs)
    if isinstance(output, (tuple, list)):
        logits = torch.squeeze(output[0], dim=-1)
    else:
        logits = output

    teacher_output = self.teacher(inputs)
    if isinstance(teacher_output, (tuple, list)):
        teacher_logits = torch.squeeze(teacher_output[0], dim=-1)
    else:
        teacher_logits = teacher_output
    
    if self.hparams.num_classes == 1:
        probs = torch.sigmoid(logits).detach().cpu()
        
        if KL_DIV:
            loss = F.kl_div(F.logsigmoid(new_logits(logits)), torch.sigmoid(new_logits(teacher_logits)), reduction="batchmean")
    else:
        probs = F.softmax(logits, dim=1).detach().cpu()
        
        if KL_DIV:
            teacher_probs = F.softmax(teacher_logits, dim=1)
            loss = F.kl_div(F.log_softmax(logits, dim=1), teacher_probs, reduction="batchmean")
     
    if not KL_DIV:
        loss = F.mse_loss(logits, teacher_logits)
    self.log("base_train_loss", loss, prog_bar=True, sync_dist=True)
        
    if REG == "aux" and TRAIN_W_REG:
        reg = LAMBDA * torch.linalg.vector_norm(teacher_output[1] - output[1], dim=1).mean()
        self.log("reg_loss", reg, prog_bar=True, sync_dist=True)
        loss += reg
    elif REG == "mlp" and TRAIN_W_REG:
        reg = LAMBDA * torch.linalg.vector_norm(outputs["teacher"] - outputs["student"], dim=1).mean()
        self.log("reg_loss", reg, prog_bar=True, sync_dist=True)
        loss += reg

    targets = targets.cpu()

    return {"loss": loss, "probs": probs, "targets": targets}

class TeacherResNet(ResNet):
    def __init__(self, args):
        super().__init__(args)
        
        if REG == "aux":
            self.model.aux = nn.Linear(self.model.fc[1].in_features, K, bias=args.bias)
            for p in self.model.aux.parameters():
                p.requires_grad = False
        elif REG == "mlp":
            self.model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout_prob, inplace=True),
                nn.Linear(self.model.fc[1].in_features, K, bias=args.bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=args.dropout_prob, inplace=True),
                nn.Linear(K, args.num_classes, bias=args.bias),
            )
            self.model.fc[1].register_forward_hook(get_output("teacher"))

    def forward(self, x):
        if TRAIN_W_REG == True and REG == "aux":
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)

            y = self.model.aux(x)
            x = self.model.fc(x)

            return (x, y)
        else:
            return super().forward(x)

class StudentResNet(ResNet):
    def __init__(self, args):
        super().__init__(args)
        
        if REG == "aux":
            self.model.aux = nn.Linear(self.model.fc[1].in_features, K, bias=args.bias)
        elif REG == "mlp":
            self.model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout_prob, inplace=True),
                nn.Linear(self.model.fc[1].in_features, K, bias=args.bias),
                nn.ReLU(inplace=True),
                nn.Dropout(p=args.dropout_prob, inplace=True),
                nn.Linear(K, args.num_classes, bias=args.bias),
            )
            self.model.fc[1].register_forward_hook(get_output("student"))

        teacher_args = deepcopy(args)
        teacher_args.resnet_version = 152
        self.teacher = TeacherResNet(teacher_args)
        state_dict = torch.load(TEACHER_WEIGHTS, map_location="cpu")["state_dict"]
        self.teacher.load_state_dict(state_dict, strict=False)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x):
        if TRAIN_W_REG == True and REG == "aux":
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)

            y = self.model.aux(x)
            x = self.model.fc(x)

            return (x, y)
        else:
            return super().forward(x)
    
    def step(self, batch, idx):
        return step_helper(self, batch)

class TeacherCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
        
        if REG == "aux":
            self.model.aux = nn.LazyLinear(K, bias=args.bias)
        elif REG == "mlp":
            self.model[-1] = nn.Sequential(
                nn.LazyLinear(K, bias=args.bias),
                nn.ReLU(inplace=True),
                nn.Linear(K, args.num_classes, bias=args.bias),
            )
            self.model[-1][0].register_forward_hook(get_output("student"))

class StudentCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
        
        if REG == "aux":
            self.model.aux = nn.LazyLinear(K, bias=args.bias)
        elif REG == "mlp":
            self.model[-1] = nn.Sequential(
                nn.LazyLinear(K, bias=args.bias),
                nn.ReLU(inplace=True),
                nn.Linear(K, args.num_classes, bias=args.bias),
            )
            self.model[-1][0].register_forward_hook(get_output("student"))
            
        teacher_args = deepcopy(args)
        teacher_args.resnet_version = 152
        self.teacher = TeacherResNet(teacher_args)
        state_dict = torch.load(TEACHER_WEIGHTS, map_location="cpu")["state_dict"]
        self.teacher.load_state_dict(state_dict, strict=False)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    # re-define forward for aux?
            
    def step(self, batch, idx):
        return step_helper(self, batch)

def experiment(args):
    if TRAIN_TEACHER:
        if args.model == "resnet":
            arch = TeacherResNet
        elif args.model == "cnn":
            arch = TeacherCNN
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
