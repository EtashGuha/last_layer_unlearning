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
SHARPNESS = [0, 0, 0]
SHARPNESS_BOUND = [0, 0, 0]
SIGMA = [0.01, 0.02, 0.05]
WIDTHS = [128, 256, 512, 1024]
NUM_MC_SAMPLES = 10


class SharpMLP(MLP):
    def __init__(self, args):
        super().__init__(args)
    
    def training_epoch_end(self, training_step_outputs):
        if self.trainer.current_epoch == self.hparams.max_epochs - 1:
            for j, sigma in enumerate(SIGMA):
                sharp = np.stack([result[f"sharp_loss_{sigma}"] for result in training_step_outputs]).mean()
                bound = np.stack([result[f"sharp_bound_{sigma}"] for result in training_step_outputs]).mean()
                SHARPNESS[j] = sharp
                SHARPNESS_BOUND[j] = bound

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)

        inputs, targets = batch
        inputs = inputs.reshape(inputs.shape[0], -1)

        if self.trainer.current_epoch == self.hparams.max_epochs - 1:
            with torch.no_grad():
                w = []
                for layer in self.model:
                    if isinstance(layer, Linear):
                        w.append(layer.weight)

                norm = torch.linalg.vector_norm(inputs, dim=1)
                r = (w[-1] ** 2).T

                for sigma in SIGMA:
                    # bound
                    # TODO: vectorize?
                    sharp_bound = []
                    for j in range(len(norm)):
                        n = norm[j]
                        v = (n * sigma) / sqrt(2 * pi) + ((n * sigma) ** 2 * (pi - 1)) / (2 * pi)
                        v = v.repeat(self.hparams.mlp_hidden_dim)
                        sharp_bound.append((v @ r).item())

                    # approx
                    sharp_loss = []
                    for _ in range(NUM_MC_SAMPLES):
                        for layer in self.model:
                            if isinstance(layer, Linear):
                                layer.weight = Parameter(torch.normal(layer.weight, sigma))
                        logits = self(inputs)
                        sharp_loss.append(F.binary_cross_entropy_with_logits(logits, targets.float()).cpu())

                        counter = 0
                        for layer in self.model:
                            if isinstance(layer, Linear):
                                layer.weight = w[counter]
                                counter += 1

                    result[f"sharp_bound_{sigma}"] = to_np(sharp_bound).mean()
                    result[f"sharp_loss_{sigma}"] = to_np(sharp_loss).mean()

        return result
            
def experiment(args):
    global SHARPNESS, SHARPNESS_BOUND

    sharpness = []
    sharpness_bound = []
    for width in WIDTHS:
        args.mlp_hidden_dim = width
        main(args, SharpMLP, BinaryMNIST)
        sharpness.append(SHARPNESS)
        sharpness_bound.append(SHARPNESS_BOUND)

    print(sharpness)
    print(sharpness_bound)

    # Plots here


if __name__ == "__main__":
    args = parse_args()
    experiment(args)

