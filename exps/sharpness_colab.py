from math import pi
import os.path as osp

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from groundzero.args import parse_args
from groundzero.main import main
from groundzero.mlp import MLP
from groundzero.utils import compute_accuracy

SIGMA = [0.02, 0.05, 0.1]
MC_SAMPLES = 10
LOSS_THRESH = 0.25
SHARP = []
SHARP_APX1 = []
SHARP_APX2 = []


def to_np(x):
    return x.cpu().detach().numpy()

class SharpnessMLP(MLP):
    def __init__(self, args, classes):
        super().__init__(args, classes)
        
        self.fc_layers = [3 * i for i in range(args.mlp_num_layers)]
        
    def training_step(self, batch, idx):
        result = super().training_step(batch, idx)

        with torch.no_grad():
            inputs, targets = batch
            inputs = inputs.reshape(inputs.shape[0], -1)
            
            w1 = self.model[0].weight
            w2 = self.model[self.fc_layers[1]].weight
            norm = torch.linalg.vector_norm(inputs, dim=1)
            
            sharp = []
            sharp_apx1 = []
            sharp_apx2 = []
            
            for sigma in SIGMA:
                avg = []
                for _ in range(MC_SAMPLES):
                    avg.append(torch.sigmoid(inputs @ torch.normal(w1, sigma).T) @ w2.T)
                logits = torch.mean(torch.stack(avg), dim=0)
                sharp.append(F.cross_entropy(logits, targets) - result["loss"])

                # maclaurin
                wx = inputs @ w1.T
                sigmoid_apx = (torch.exp(-wx) + 1) ** -1
                sigmoid_apx += ((torch.exp(-wx) - 1) * torch.exp(-wx)) / (2 * (torch.exp(-wx) + 1) ** 3) * ((sigma * norm) ** 2).view(-1, 1)
                sigmoid_apx += (((torch.exp(-3 * wx) - 11 * torch.exp(-2 * wx) + 11 * torch.exp(-wx) -1) * torch.exp(-wx) )/ (8 * (torch.exp(-wx) + 1) ** 5)) * ((sigma * norm) ** 4).view(-1, 1)
                logits_apx = sigmoid_apx @ w2.T
                sharp_apx1.append(F.cross_entropy(logits_apx, targets) - result["loss"])

                # probit
                cdf = torch.distributions.normal.Normal(0,1).cdf
                denom = torch.sqrt(torch.tensor(8/pi) + (sigma * norm) ** 2)
                sigmoid_apx = cdf(wx / denom.view(-1, 1))
                logits_apx = sigmoid_apx @ w2.T
                sharp_apx2.append(F.cross_entropy(logits_apx, targets) - result["loss"])
            
            result["sharp"] = sharp
            result["sharp_apx1"] = sharp_apx1
            result["sharp_apx2"] = sharp_apx2

        return result
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)
        
        sharp = to_np(torch.stack([result["sharp"] for result in training_step_outputs]))
        sharp_apx1 = to_np(torch.stack([result["sharp_apx1"] for result in training_step_outputs]))
        sharp_apx2 = to_np(torch.stack([result["sharp_apx2"] for result in training_step_outputs]))
        
        SHARP.append(torch.mean(sharp, dim=1))
        SHARP_APX1.append(torch.mean(sharp_apx1, dim=1))
        SHARP_APX2.append(torch.mean(sharp_apx2, dim=1))
        
def experiment(args):
    global SHARP, SHARP_APX1, SHARP_APX2
                      
    callbacks = [
        EarlyStopping(
            monitor="train_loss",
            stopping_threshold=LOSS_THRESH,
        ),
    ]
    
    args.num_sanity_val_steps = 0
    main(args, SharpnessMLP, callbacks=callbacks)
    
    SHARP = np.asarray(SHARP)
    SHARP_APX1 = np.asarray(SHARP_APX1)
    SHARP_APX2 = np.asarray(SHARP_APX2)
    
    x = np.arange(len(SHARP))
    
    for j, sigma in enumerate(SIGMA):
        sharp = SHARP[:, j]
        sharp_apx1 = SHARP_APX1[:, j]
        sharp_apx2 = SHARP_APX2[:, j]
        
        plt.plot(x, sharp, label=f"Actual - {MC_SAMPLES} MC samples", linestyle="solid")
        plt.plot(x, sharp_apx1, label="Maclaurin - 3 terms", linestyle="dashed")
        plt.plot(x, sharp_apx2, label="Probit", linestyle="dotted")
        plt.xlabel("Epoch")
        plt.ylabel("Sharpness")
        plt.legend()
        plt.title(f"MNIST, SGD 0.05, SIGMA {sigma}, WD 0, B 256, LOSS {LOSS_THRESH}")
        plt.savefig(osp.join(args.out_dir, f"sharpness_sigma{sigma}.png"))
        plt.clf()      


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
