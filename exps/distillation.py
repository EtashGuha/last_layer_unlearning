from copy import deepcopy

from groundzero.args import parse_args
from groundzero.datamodules import Waterbirds
from groundzero.main import main
from groundzero.models import ResNet

TEACHER_WEIGHTS = ""


class DistillationResNet(ResNet):
    def __init__(self, args):
        super().__init__(args)

        # Possibly cache teacher predictions.
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

        if self.training:
            targets = self.teacher(inputs)

        logits = self(inputs)

        if self.hparams.num_classes == 1:
            if self.hparams.loss == "mse":
                loss = F.mse_loss(logits, targets)
                probs = logits.detach().cpu()
            else:
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                probs = torch.sigmoid(logits).detach().cpu()
        else:
            if self.hparams.loss == "mse":
                return ValueError("MSE is only an option for binary classification.")
            loss = F.cross_entropy(logits, targets)
            probs = F.softmax(logits, dim=1).detach().cpu()

        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}

def experiment(args):
    main(args, DistillationResNet, Waterbirds)


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
