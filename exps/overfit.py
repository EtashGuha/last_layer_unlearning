from math import pi
import os.path as osp
import pickle

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from groundzero.args import parse_args
from groundzero.cnn import CNN
from groundzero.main import main
from groundzero.utils import compute_accuracy, to_np

ACCS = [0, 0, 0]
LOSS_THRESH = 0.01
WIDTHS = [16, 32, 64]


class OverfitCNN(CNN):
    def __init__(self, args, classes):
        super().__init__(args, classes)
        
        self.j = args.j
    
    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        
        ACCS[self.j] = self.val_acc1

def experiment(args):
    global ACCS
                      
    callbacks = [
        EarlyStopping(
            monitor="train_loss",
            stopping_threshold=LOSS_THRESH,
        ),
    ]
    
    main(args, OverfitCNN, callbacks=callbacks)
    
    for j, sigma in enumerate(WIDTHS):
        x = np.arange(len(sharp))
        plt.plot(x, sharp, label=f"Actual - {MC_SAMPLES} MC samples")
        plt.plot(x, sharp_apx1, label="Maclaurin - degree 4")
        plt.plot(x, sharp_apx2, label="Probit")
        plt.xlabel("Step (Moving Avg)")
        plt.ylabel("Sharpness")
        plt.legend()
        plt.title(f"MNIST, (w: {args.mlp_hidden_dim}, d: {args.mlp_num_layers}), SGD 0.05, SIGMA {sigma}, B 256, LOSS {LOSS_THRESH}")
        plt.savefig(osp.join(args.out_dir, f"sharpness_sigma{sigma}.png"))
        plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
