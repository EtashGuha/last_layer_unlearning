import os.path as osp

import torch
from torch.fft import fft2
from torch.linalg import svdvals
from torch.nn import Conv2d

import matplotlib.pyplot as plt
import numpy as np

from groundzero.args import parse_args
from groundzero.datasets.cifar10 import CIFAR10
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.utils import to_np

WIDTHS = [2, 4, 6, 8, 10]


class OverfitCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
        
    def test_epoch_end(self, test_step_outputs):
        super().test_epoch_end(test_step_outputs)
        
        top_svs = []
        for layer in self.model:
            if isinstance(layer, Conv2d):
                transforms = fft2(layer.weight)
                svs = svdvals(transforms)
                top_svs.append(torch.max(svs).item())
        
        test_prod_spec = np.prod(np.asarray(top_svs))
        self.log("prod_spec", test_prod_spec)
            
def experiment(args):
    accs = []
    norms = []
    for width in WIDTHS:
        args.cnn_initial_width = width
        result = main(args, OverfitCNN, CIFAR10)
        accs.append(result[0]["test_acc1"])
        norms.append(result[0]["prod_spec"])

    print(accs)
    print(norms)

    accs = 1 - np.asarray(accs)
    plt.plot(WIDTHS, accs)
    plt.xlabel("CNN Width Parameter")
    plt.ylabel("Test Error")
    plt.legend()
    plt.title(f"CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.max_epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit.png"))
    plt.clf()

    plt.plot(WIDTHS, norms)
    plt.xlabel("Product of Conv2D Spectral Norms")
    plt.ylabel("Test Error")
    plt.legend()
    plt.title(f"CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.max_epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit2.png"))
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
