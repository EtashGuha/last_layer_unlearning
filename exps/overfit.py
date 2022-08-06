import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from groundzero.args import parse_args
from groundzero.cnn import CNN
from groundzero.main import main

LOSS_THRESH = 0.1
TRAIN_PROPORTION = 0.1
WIDTHS = [16, 32, 64]


def experiment(args):
    callbacks = [
        EarlyStopping(
            monitor="train_loss",
            patience=1000,
            stopping_threshold=LOSS_THRESH,
        ),
    ]
    
    args.limit_train_batches = TRAIN_PROPORTION
    args.check_val_every_n_epoch = int(1 / TRAIN_PROPORTION)
    
    accs = []
    for j, width in enumerate(WIDTHS):
        args.cnn_initial_width = width
        args.j = j
        
        result = main(args, CNN, callbacks=callbacks)
        accs.append(result["acc1"])

    plt.plot(WIDTHS, 1-np.asarray(accs))
    plt.xlabel("CNN Width Parameter")
    plt.ylabel("Test Error")
    plt.legend()
    plt.title(f"Subsampled {TRAIN_PROPORTION} CIFAR-10, {args.cnn_num_layers} layer CNN, SGD 0.05, B 256, LOSS {LOSS_THRESH}")
    plt.savefig(osp.join(args.out_dir, f"overfit.png"))
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
