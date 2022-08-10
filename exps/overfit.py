import os.path as osp

import torch
from torch.fft import fft2
from torch.linalg import svdvals
from torch.nn import Conv2d, Linear

import matplotlib.pyplot as plt
import numpy as np

from groundzero.args import parse_args
from groundzero.datasets.cifar10 import CIFAR10
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.utils import to_np

TRAIN_ACC1 = [0]
WIDTHS = [2, 4, 6, 8, 10]


class OverfitCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)
        TRAIN_ACC1[0] = self.train_acc1
    
    def test_epoch_end(self, test_step_outputs):
        super().test_epoch_end(test_step_outputs)
        
        top_svs = []
        for layer in self.model:
            if isinstance(layer, Conv2d):
                transforms = fft2(layer.weight)
                svs = svdvals(transforms)
                top_svs.append(torch.max(svs).item())
            elif isinstance(layer, Linear):
                svs = svdvals(layer.weight)
                top_svs.append(torch.max(svs).item())
        
        prod_spec = np.prod(np.asarray(top_svs))
        self.log("prod_spec", prod_spec)
            
def experiment(args):
    global TRAIN_ACC1
    
    train_accs = []
    test_accs = []
    norms = []
    for width in WIDTHS:
        args.cnn_initial_width = width
        result = main(args, OverfitCNN, CIFAR10)
        train_accs.append(TRAIN_ACC1[0])
        test_accs.append(result[0]["test_acc1"])
        norms.append(result[0]["prod_spec"])

    print(train_accs)
    print(test_accs)
    print(norms)

    errs = 1 - np.asarray(test_accs)
    
    plt.plot(WIDTHS, errs)
    plt.xlabel("CNN Width Parameter")
    plt.ylabel("Test Error")
    plt.title(f"CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.max_epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit.png"))
    plt.clf()

    plt.plot(norms, errs)
    plt.xlabel("Product of Conv2D Spectral Norms")
    plt.ylabel("Test Error")
    plt.title(f"CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.max_epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit2.png"))
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
