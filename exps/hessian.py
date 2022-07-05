import os.path as osp

import torch
import torch.nn.functional as F

from hessian_eigenthings import compute_hessian_eigenthings
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from testbed.args import parse_args
from testbed.main import main
from testbed.resnet import ResNet

EIGENVALS = []
SMALLBATCH_SIZE = 128
LARGEBATCH_SIZE = 4096
TRAIN_ACC_STOP_THRESHOLD = 0.99


class HessianResNet(ResNet):
    def __init__(self, args, classes):
        super().__init__(args, classes)

    def validation_epoch_end(self, validation_step_outputs):
        if self.train_acc1 >= TRAIN_ACC_STOP_THRESHOLD:
            torch.set_grad_enabled(True)
            loss = F.cross_entropy
            loader = self.trainer.val_dataloaders[0]
            eigenval = compute_hessian_eigenthings(self.model, loader, loss, 1)[0][0]
            EIGENVALS.append(eigenval)

def experiment(args):
    args.max_epochs = 1000
    lrs = [0.001, 0.005, 0.01, 0.05]


    callbacks = [EarlyStopping(
                    monitor="train_acc1",
                    mode="max",
                    stopping_threshold=TRAIN_ACC_STOP_THRESHOLD,
                    check_on_train_epoch_end=True,
                )]

    for lr in lrs:
        args.lr = lr
        args.batch_size = SMALLBATCH_SIZE
        main(args, HessianResNet, callbacks=callbacks)

    print(EIGENVALS)

    for lr in lrs:
        args.lr = lr
        args.batch_size = LARGEBATCH_SIZE
        main(args, HessianResNet, callbacks=callbacks)

    print(EIGENVALS)


if __name__ == "__main__":
    args = parse_args()
    experiment(args)

