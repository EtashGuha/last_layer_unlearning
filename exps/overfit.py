import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from groundzero.args import parse_args
from groundzero.datasets.cifar10 import CIFAR10
from groundzero.main import main
from groundzero.models.cnn import CNN

DEPTHS = [2, 4]
WIDTHS = [32, 64, 128, 256]


def experiment(args):
    accs = []
    for depth in DEPTHS:
        a = []
        args.cnn_num_layers = depth
        for width in WIDTHS:
            args.cnn_initial_width = width
            result = main(args, CNN, CIFAR10)
            a.append(result[0]["test_acc1"])
        accs.append(a)

    accs = 1 - np.asarray(accs)
    plt.plot(WIDTHS, accs[0], label="2 layer CNN")
    plt.plot(WIDTHS, accs[1], label="4 layer CNN")
    plt.xlabel("CNN Width Parameter")
    plt.xscale("log", base=2) 
    plt.ylabel("Test Error")
    plt.legend()
    plt.title(f"CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.max_epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit.png"))
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
