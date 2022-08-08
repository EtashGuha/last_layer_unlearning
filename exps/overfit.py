import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from groundzero.args import parse_args
from groundzero.nin import NiN
from groundzero.main import main

TRAIN_PROPORTION = 0.1
DEPTHS = [2, 4]
WIDTHS = [96 * 1, 96 * 2, 96 * 3, 96 * 4]


def experiment(args):
    args.limit_train_batches = TRAIN_PROPORTION
    
    accs = []
    for depth in DEPTHS:
        a = []
        args.nin_num_layers = depth
        for width in WIDTHS:
            args.nin_width = width
            result = main(args, NiN)
            a.append(result[0]["test_acc1"])
        accs.append(a)

    accs = 1 - np.asarray(accs)
    plt.plot(WIDTHS, accs[0], label="3 layer NiN")
    plt.plot(WIDTHS, accs[1], label="5 layer NiN")
    plt.xlabel("NiN Width Parameter")
    plt.xscale("log", base=96) 
    plt.ylabel("Test Error")
    plt.legend()
    plt.title(f"Subsampled {TRAIN_PROPORTION} CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.max_epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit.png"))
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
