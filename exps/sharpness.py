from copy import deepcopy
from math import pi, sqrt

import torch
from torch.nn import Linear, Parameter
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from groundzero.args import parse_args
from groundzero.datasets.binary_mnist import BinaryMNIST
from groundzero.main import main
from groundzero.models.mlp import MLP
from groundzero.utils import to_np

#NOISE = [0]
SIGMA = [0.01, 0.05, 0.5, 1, 2, 5]
SHARPNESS = [0 for _ in SIGMA]
SHARPNESS_BOUND = [0 for _ in SIGMA]
#WIDTHS = [128, 256, 512, 1024]
#WIDTHS = [128, 256]
WIDTHS = [128, 256]
WEIGHTS = [257, 266]
#WEIGHTS = [234, 235]
#WEIGHTS = [None, None]
NUM_MC_SAMPLES = 10


class SharpMLP(MLP):
    def __init__(self, args):
        super().__init__(args)

        if WEIGHTS[0]:
            self.automatic_optimization = False
    
    def training_epoch_end(self, training_step_outputs):
        # make last layer weights positive
        with torch.no_grad():
            self.model[-1].weight.copy_(self.model[-1].weight.data.clamp(min=0))

        if self.trainer.current_epoch == self.hparams.max_epochs - 1:
            for j, sigma in enumerate(SIGMA):
                sharp = np.stack([result[f"sharp_loss_{sigma}"] for result in training_step_outputs]).mean()
                bound = np.stack([result[f"sharp_bound_{sigma}"] for result in training_step_outputs]).mean()
                SHARPNESS[j] = sharp
                SHARPNESS_BOUND[j] = bound

    def training_step(self, batch, batch_idx):
        # make last layer weights positive
        # TODO: Could subclass MLP to be a little cleaner + work w/ backprop.
        with torch.no_grad():
            self.model[-1].weight.copy_(self.model[-1].weight.clamp(min=0))

        result = super().training_step(batch, batch_idx)

        inputs, targets = batch
        inputs = inputs.reshape(inputs.shape[0], -1)

        if self.trainer.current_epoch == self.hparams.max_epochs - 1:
            with torch.no_grad():
                # 2 layers only
                w = deepcopy(self.model[0].weight)
                #r = (self.model[-1].weight ** 2).T
                w2_l1_norm_sq = torch.linalg.vector_norm(self.model[-1].weight, ord=1) ** 2
                w2_l2_norm_sq = torch.linalg.vector_norm(self.model[-1].weight, ord=2) ** 2
                data_2norm_sq = (torch.linalg.vector_norm(inputs, dim=1) ** 2)
                #sum_data_2norm_sq = (torch.linalg.vector_norm(inputs, dim=1) ** 2).sum()

                for sigma in SIGMA:
                    # bound
                    """
                    v = (norm * sigma) / sqrt(2 * pi) + ((norm * sigma) ** 2 * (pi - 1)) / (2 * pi)
                    v = v.unsqueeze(1).repeat(1, self.hparams.mlp_hidden_dim)
                    sharp_bound = (v @ r).mean().item()
                    """
                    """
                    print(f"w2_l1_norm: {w2_l1_norm}")
                    print(f"sigma: {sigma}")
                    print(f"m: {len(inputs)}")
                    print(f"sum_data_2norm_sq: {sum_data_2norm_sq}")
                    sharp_bound = (((w2_l1_norm * sigma) ** 2) / (2 * len(inputs)) * sum_data_2norm_sq).item()
                    sharp_bound = ((sigma ** 2 * (w2_l1_norm ** 2 + (pi - 1) * w2_l2_norm ** 2)) / (2 * pi * len(inputs)) * sum_data_2norm_sq).item()
                    """
                    # general ver.
                    sharp_bound = ((sigma ** 2 * data_2norm_sq.mean()) * (w2_l1_norm_sq / (2 * pi) + w2_l2_norm_sq)).item()
                    # (slightly) tighter ver.
                    #q = inputs @ w.T
                    #q = (q > 0).float()
                    #q = (self.model[-1].weight ** 2) @ q.T
                    #sharp_bound = ( sigma ** 2 * (data_2norm_sq * ((w2_l1_norm_sq / (2 * pi)) + q)).mean() ).item()

                    # MC approx
                    # note: can subtract true loss but assuming ~0
                    sharp_loss = []
                    for _ in range(NUM_MC_SAMPLES):
                        self.model[0].weight.copy_(torch.normal(w, sigma))
                        logits = self(inputs)
                        sharp_loss.append(F.mse_loss(logits, targets.float()).cpu())
                    sharp_loss = to_np(sharp_loss).mean()

                    result[f"sharp_bound_{sigma}"] = sharp_bound
                    result[f"sharp_loss_{sigma}"] = sharp_loss

                self.model[0].weight.copy_(w)

        return result
            
def experiment(args):
    global SHARPNESS, SHARPNESS_BOUND

    sharpness = []
    sharpness_bound = []
    for weights, width in zip(WEIGHTS, WIDTHS):
        if weights:
            args.weights = f"lightning_logs/version_{weights}/checkpoints/last.ckpt"
            args.max_epochs = 1
        args.mlp_hidden_dim = width
        main(args, SharpMLP, BinaryMNIST)
        sharpness.append(deepcopy(SHARPNESS))
        sharpness_bound.append(deepcopy(SHARPNESS_BOUND))

    print(sharpness)
    print(sharpness_bound)

    # Plots here


if __name__ == "__main__":
    args = parse_args()
    experiment(args)

