import torch
from torch.nn import Linear, Parameter
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from groundzero.args import parse_args
from groundzero.datasets.mnist import MNIST
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
            loss = torch.stack([result["loss"] for result in training_step_outputs]).mean().item()
            for j, sigma in enumerate(SIGMA):
                sharp = np.stack([result["sharp_loss"] for result in training_step_outputs]).mean()
                SHARPNESS[j] = sharp - loss

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        result = super().training_step(batch, batch_idx)

        if self.trainer.current_epoch == self.hparams.max_epochs - 1:
            with torch.no_grad():
                w = []
                for layer in self.model:
                    if isinstance(layer, Linear):
                        w.append(layer.weight)

                sharp_loss = []
                for _ in range(NUM_MC_SAMPLES):
                    for layer in self.model:
                        if isinstance(layer, Linear):
                            layer.weight = Parameter(torch.normal(layer.weight, SIGMA))
                    logits = self(inputs)
                    sharp_probs = F.softmax(logits, dim=1).detach()
                    sharp_loss.append(F.cross_entropy(sharp_probs, targets).cpu())

                    counter = 0
                    for layer in self.model:
                        if isinstance(layer, Linear):
                            layer.weight = w[counter]
                            counter += 1

                result["sharp_loss"] = to_np(sharp_loss).mean()

        return result
            
def experiment(args):
    global SHARPNESS, SHARPNESS_BOUND

    sharpness = []
    sharpness_bound = []
    for width in WIDTHS:
        args.mlp_width = width
        main(args, SharpMLP, MNIST)
        sharpness.append(SHARPNESS)
        sharpness_bound.append(SHARPNESS_BOUND)

    print(sharpness)
    print(sharpness_bound)

    # Plots here


if __name__ == "__main__":
    args = parse_args()
    experiment(args)

