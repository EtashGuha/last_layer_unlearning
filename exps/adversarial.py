from copy import deepcopy
from time import time

from pytorch_lightning import Trainer
import torch
import torch.nn.functional as F

from configargparse import Parser
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

from groundzero.args import add_input_args
from groundzero.datamodules.cifar10 import CIFAR10
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.models.resnet import ResNet
from groundzero.utils import to_np


class PGDAttack():
    def __init__(self, args, model):
        self.model = model

        try:
            self.alpha = args.alpha
            self.epsilon = args.epsilon
            self.pgd_steps = args.pgd_steps
        except:
            self.alpha = args["alpha"]
            self.epsilon = args["epsilon"]
            self.pgd_steps = args["pgd_steps"]

    def perturb(self, x_natural, y, compute_fosc=False):
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

            fosc = None
            if self.pgd_steps > 0 and compute_fosc:
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.requires_grad_()
                grad = torch.autograd.grad(loss, [x])[0]

                grad = to_np(grad).reshape(len(x), -1)
                grad_norm = LA.norm(grad, ord=1, axis=1).reshape(-1, 1)
                diff = to_np(x - x_natural).reshape(len(x), -1)
                fosc = np.copy(grad_norm)
                for i in range(len(x)):
                    fosc[i] = max(0., - np.dot(grad[i], diff[i]) + self.epsilon * grad_norm[i])
                fosc = fosc.mean()

        return x, fosc

class AdversarialCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
        self.adversary = PGDAttack(args, self.model)
        self.fosc = []

    def step(self, batch, idx):
        inputs, targets = batch

        if not self.training:
            self.hparams["pgd_steps"] = 20
            self.adversary = PGDAttack(self.hparams, self.model)
            adv, _ = self.adversary.perturb(inputs, targets)
            self.hparams["pgd_steps"] = 0
        elif self.hparams["pgd_steps"] and self.trainer.current_epoch == self.hparams["max_epochs"] - 1:
            adv, fosc = self.adversary.perturb(inputs, targets, compute_fosc=True)
            self.fosc.append(fosc)
        elif self.hparams["pgd_steps"]:
            adv, _ = self.adversary.perturb(inputs, targets)
        else:
            adv = inputs

        adv_outputs = self.model(adv)
        loss = F.cross_entropy(adv_outputs, targets)
        probs = F.softmax(adv_outputs, dim=1).detach().cpu()
        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}


class AdversarialResNet(ResNet):
    def __init__(self, args):
        super().__init__(args)
        self.adversary = PGDAttack(args, self.model)
        self.fosc = []

    def step(self, batch, idx):
        inputs, targets = batch

        if self.trainer.current_epoch == self.hparams["max_epochs"] - 1:
            adv, fosc = self.adversary.perturb(inputs, targets, compute_fosc=True)
            self.fosc.append(fosc)
        else:
            adv, _ = self.adversary.perturb(inputs, targets)

        adv_outputs = self.model(adv)
        loss = F.cross_entropy(adv_outputs, targets)
        probs = F.softmax(adv_outputs, dim=1).detach().cpu()
        targets = targets.cpu()

        return {"loss": loss, "probs": probs, "targets": targets}


def experiment(args):
    args.dropout_prob = 0
    args.max_epochs = 100
    args.lr_steps = [50, 75]

    if args.model == "cnn":
        STEPS = [0, 5, 10, 20]
        WIDTHS = [16, 32, 64, 128, 256]

        # ERM baseline
        accs = []
        for width in WIDTHS:
            args.cnn_initial_width = width
            _, _, test_metrics = main(args, CNN, CIFAR10)
            accs.append(test_metrics[0]["test_acc1"])
        plt.plot(WIDTHS, accs)
        plt.xlabel("CNN initial width")
        plt.ylabel("Accuracy")
        plt.xscale("log", base=2)
        plt.savefig("erm.png", bbox_inches="tight")
        plt.clf()

        # Adversarial training
        accs = []
        foscs = []
        times = []
        for step in STEPS:
            step_acc = []
            step_fosc = []
            step_times = []
            args.pgd_steps = step
            for width in WIDTHS:
                args.cnn_initial_width = width
                start = time()
                model, _, test_metrics = main(args, AdversarialCNN, CIFAR10)
                end = time()
                step_acc.append(test_metrics[0]["test_acc1"])
                if step != 0:
                    step_fosc.append(to_np(model.fosc).mean())
                step_times.append(end-start)
            accs.append(step_acc)
            foscs.append(step_fosc)
            times.append(step_times)
        print(accs)
        print(foscs)
        print(times)

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

