from configargparse import Parser
from copy import deepcopy

from pytorch_lightning import Trainer
import torch
import torch.nn.functional as F

from groundzero.args import add_input_args
from groundzero.datamodules.cifar10 import CIFAR10
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.models.resnet import ResNet

class PGDAttack():
    def __init__(self, args, model):
        self.model = model

        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.pgd_steps = args.pgd_steps

    def perturb(self, x_natural, y):
        with torch.inference_mode(False):
            y = torch.clone(y)
            x = x_natural.detach()
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            for i in range(self.pgd_steps):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.requires_grad_()
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.alpha * torch.sign(grad.detach())
                x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
                x = torch.clamp(x, 0, 1)
        return x

class AdversarialCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
        self.adversary = PGDAttack(args, self.model)

    def step(self, batch, idx):
        input, targets = batch

        adv = self.adversary.perturb(input, targets)
        adv_outputs = self.model(adv)

        loss = F.cross_entropy(adv_outputs, targets)
        probs = F.softmax(adv_outputs, dim=1).detach().cpu()
        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}


class AdversarialResNet(ResNet):
    def __init__(self, args):
        super().__init__(args)
        self.adversary = PGDAttack(args, self.model)

    def step(self, batch, idx):
        input, targets = batch

        adv = self.adversary.perturb(input, targets)
        adv_outputs = self.model(adv)

        loss = F.cross_entropy(adv_outputs, targets)
        probs = F.softmax(adv_outputs, dim=1).detach().cpu()
        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}


def experiment(args):
    args.dropout_prob = 0
    args.max_epochs = 90
    args.lr_steps = [30, 60]

    if args.model == "cnn":
        """
        WIDTHS = [32]

        for width in WIDTHS:
            args.cnn_initial_width = width
            main(args, AdversarialCNN, CIFAR10)
        """
        main(args, AdversarialCNN, CIFAR10)

    elif args.model == "resnet":
        main(args, AdversarialResNet, CIFAR10)

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add("--alpha", default=0.00784, type=float)
    parser.add("--epsilon", default=0.0314, type=float)
    parser.add("--pgd_steps", default=7, type=int)

    args = parser.parse_args()
    experiment(args)

