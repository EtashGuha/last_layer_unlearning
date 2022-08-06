import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from groundzero.args import parse_args
from groundzero.cnn import CNN
from groundzero.main import main

TRAIN_PROPORTION = 0.1
DEPTHS = [2, 4]
WIDTHS = [32, 64, 128]


def experiment(args):
    args.limit_train_batches = TRAIN_PROPORTION
    
    accs = []
    for depth in DEPTHS:
        a = []
        args.cnn_num_layers = depth
        for width in WIDTHS:
            args.cnn_initial_width = width
            result = main(args, CNN)
            a.append(result[0]["test_acc1"])
        accs.append(a)

    accs = 1 - np.asarray(accs)
    plt.plot(WIDTHS, accs[0], label="3 layer CNN")
    plt.plot(WIDTHS, accs[1], label="5 layer CNN")
    plt.xlabel("CNN Width Parameter")
    plt.ylabel("Test Error")
    plt.legend()
    plt.title(f"Subsampled {TRAIN_PROPORTION} CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit.png"))
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
