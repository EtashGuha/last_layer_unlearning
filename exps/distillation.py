from copy import deepcopy

import torch
import torch.nn.functional as F

from groundzero.args import parse_args
from groundzero.datamodules.waterbirds import Waterbirds
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.models.resnet import ResNet

TRAIN_TEACHER = False
REG = None # None, "mlp", "aux"
K = 32
LAMBDA = 0.01
KL_DIV = False
TEACHER_WEIGHTS = "lightning_logs/version_0/checkpoints/epoch=00-val_loss=0.292-val_acc1=0.900.ckpt"
#TEACHER_WEIGHTS = "lightning_logs/version_0/checkpoints/last.ckpt"


def new_logits(logits):
    col1 = logits.unsqueeze(1)
    col2 = -logits.unsqueeze(1)

    return torch.cat((col1, col2), 1)

def step_helper(self, batch):
    inputs, targets = batch

    logits = self(inputs)
    teacher_logits = self.teacher(inputs)
    
    if self.hparams.num_classes == 1:
        probs = torch.sigmoid(logits).detach().cpu()
        
        if KL_DIV:
            loss = F.kl_div(F.logsigmoid(new_logits(logits)), F.sigmoid(new_logits(teacher_logits)), reduction="batchmean")
    else:
        probs = F.softmax(logits, dim=1).detach().cpu()
        
        if KL_DIV:
            teacher_probs = F.softmax(teacher_logits, dim=1)
            loss = F.kl_div(F.log_softmax(logits, dim=1), teacher_probs, reduction="batchmean")
     
    if not KL_DIV:
        loss = F.mse_loss(logits, teacher_logits)
        
    if REG == "aux":
        raise NotImplementedError()
    elif REG == "mlp":
        """ This copying is not correct. Do a forward hook instead: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5 """
        orig_teacher = deepcopy(self.teacher.model.fc)
        self.teacher.model.fc = self.teacher.model.fc[0]
        
        if type(self) == CNN:
            orig_student = deepcopy(self.model[-1])
            self.model[-1] = self.model[-1][0]
        else:
            orig_student = deepcopy(self.model.fc)
            self.model.fc = self.model.fc[0]
            
        k = self(inputs)
        teacher_k = self.teacher(inputs)

        loss += LAMBDA * torch.linalg.vector_norm(k - teacher_k, dim=1)
        
        self.teacher.model.fc = orig_teacher
        if type(self) == CNN:
            self.model[-1] = orig_student
        else:
            self.model.fc = orig_student

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
            fc1 = nn.Sequential(
                nn.Dropout(p=args.dropout_prob, inplace=True),
                nn.Linear(self.model.fc[1].in_features, K, bias=args.bias),
            )
            fc2 = nn.Sequential(
                nn.Dropout(p=args.dropout_prob, inplace=True),
                nn.Linear(K, args.num_classes, bias=args.bias),
            )
            self.model.fc = nn.Sequential(fc1, fc2)

class StudentResNet(ResNet):
    def __init__(self, args):
        super().__init__(args)
        
        if REG == "aux":
            self.model.aux = nn.Linear(self.model.fc[1].in_features, K, bias=args.bias)
        elif REG == "mlp":
            fc1 = nn.Sequential(
                nn.Dropout(p=args.dropout_prob, inplace=True),
                nn.Linear(self.model.fc[1].in_features, K, bias=args.bias),
            )
            fc2 = nn.Sequential(
                nn.Dropout(p=args.dropout_prob, inplace=True),
                nn.Linear(K, args.num_classes, bias=args.bias),
            )
            self.model.fc = nn.Sequential(fc1, fc2)

        teacher_args = deepcopy(args)
        teacher_args.resnet_version = 152
        self.teacher = ResNet(teacher_args)
        state_dict = torch.load(TEACHER_WEIGHTS, map_location="cpu")["state_dict"]
        self.teacher.load_state_dict(state_dict, strict=False)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    # re-define forward for aux?
    
    def step(self, batch, idx):
        return step_helper(self, batch)

class StudentCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
        
        if REG == "aux":
            self.model.aux = nn.LazyLinear(K, bias=args.bias)
        elif REG == "mlp":
            fc1 = nn.LazyLinear(K, bias=args.bias)
            fc2 = nn.Linear(K, args.num_classes, bias=args.bias)
            self.model[-1] = nn.Sequential(fc1, fc2)
            
        teacher_args = deepcopy(args)
        teacher_args.resnet_version = 152
        self.teacher = ResNet(teacher_args)
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
        arch = TeacherResNet
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
