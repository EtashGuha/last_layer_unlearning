import os.path as osp

import torch
import torch.nn.functional as F

from hessian_eigenthings import compute_hessian_eigenthings
import matplotlib.pyplot as plt

from testbed.args import parse_args
from testbed.main import main
from testbed.resnet import ResNet

class ResNetExp(ResNet):
    def __init__(self, args, classes):
        super().__init__(args, classes)

    def validation_epoch_end(self, validation_step_outputs):
        if self.current_epoch == 0:
            torch.set_grad_enabled(True)
            loss = F.cross_entropy
            loader = self.trainer.val_dataloaders[0]
            print(compute_hessian_eigenthings(self.model, loader, loss, 1)[0])

if __name__ == "__main__":
    args = parse_args()
    main(args, model_class=ResNetExp)

