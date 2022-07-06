from copy import deepcopy
import os.path as osp

import torch
import torch.nn.functional as F

from hessian_eigenthings import compute_hessian_eigenthings
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from testbed.args import parse_args
from testbed.main import main
from testbed.resnet import ResNet

RHO = 0.05
EIGENVALS = []
SHARPNESSES = []
SMALLBATCH_SIZE = 128
LARGEBATCH_SIZE = 4096
TRAIN_ACC_STOP_THRESHOLD = 0.99


class HessianResNet(ResNet):
    def __init__(self, args, classes):
        super().__init__(args, classes)

    def validation_epoch_end(self, validation_step_outputs):
        if self.train_acc1 >= TRAIN_ACC_STOP_THRESHOLD:
            # Computes maximum Hessian eigenvalue
            torch.set_grad_enabled(True)
            loss = F.cross_entropy
            loader = self.trainer.val_dataloaders[0]
            eigenval = compute_hessian_eigenthings(self.model, loader, loss, 1)[0][0]
            EIGENVALS.append(eigenval)
            torch.set_grad_enabled(False) 
            print(EIGENVALS)

            # Computes worst-case sharpness viz. SAM
            params = deepcopy(self.model.parameters())
            for p in self.model.parameters():
                epsilon = RHO * F.normalize(p.grad)
                p = p + epsilon

            losses = []
            sam_losses = []
            for j, step in enumerate(validation_step_outputs):
                batch = (step["imgs"], step["targets"])
                losses.append(step["loss"])
                sam_losses.append(self.step(batch, j)["loss"])
            losses = torch.stack(losses)
            sam_losses = torch.stack(sam_losses)
            sharpness = sam_losses.mean() - losses.mean()
            SHARPNESSES.append(sharpness)

            self.model.parameters = params

def experiment(args):
    args.max_epochs = 1000
    #lrs = [0.001, 0.005, 0.01, 0.05]
    lrs = [0.05]

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

    """
    for lr in lrs:
        args.lr = lr
        args.batch_size = LARGEBATCH_SIZE
        main(args, HessianResNet, callbacks=callbacks)

    print(EIGENVALS)
    print(SHARPNESSSES)
    """


if __name__ == "__main__":
    args = parse_args()
    experiment(args)

