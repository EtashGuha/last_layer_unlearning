import os.path as osp

import torch
from torch.autograd.functional import hessian
from torch.nn.utils import _stateless

import matplotlib.pyplot as plt

from testbed.args import parse_args
from testbed.main import main
from testbed.resnet import ResNet

class ResNetExp(ResNet):
    def __init__(self, args, classes):
        super().__init__(args, classes)

    def validation_epoch_end(self, validation_step_outputs):
        imgs = torch.stack([result["imgs"] for result in validation_step_outputs])
        targets = torch.stack([result["targets"] for result in validation_step_outputs])
        batch = (imgs, targets)

        names = list(n for n, _ in self.model.named_parameters())
        def loss(*params):
            result: torch.Tensor = _stateless.functional_call(self.step, {n: p for n, p in zip(names, params)}, batch, 0)
            return result["loss"]

        if self.current_epoch == 1:
            print(hessian(loss, tuple(self.model.parameters())))

            """
            plt.hist([param.weight for param in self.model.parameters()], 50)
            plt.xlabel("Weight")
            plt.ylabel("Frequency")
            plt.title(f"ResNet CIFAR-10 Epoch {self.current_epoch} Weight Distribution")
            plt.savefig(osp.join(self.hparams.out_dir, f"weights{self.current_epoch}.png")
            """

if __name__ == "__main__":
    args = parse_args()
    main(args, model_class=ResNetExp)

