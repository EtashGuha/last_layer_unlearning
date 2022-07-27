from math import pi
import os.path as osp
import pickle

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from groundzero.args import parse_args
from groundzero.main import main
from groundzero.mlp import MLP
from groundzero.utils import compute_accuracy, to_np

SIGMA = [0.01, 0.02, 0.05]
MC_SAMPLES = 10
MOVING_AVG = 20
LOSS_THRESH = 0.25
SHARP = []
SHARP_APX1 = []
SHARP_APX2 = []
SHARP_APX3 = []


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
            sharp_apx3 = []
            
            for sigma in SIGMA:
                avg = []
                for _ in range(MC_SAMPLES):
                    avg.append(torch.sigmoid(inputs @ torch.normal(w1, sigma).T) @ w2.T)
                logits = torch.mean(torch.stack(avg), dim=0)
                val = F.cross_entropy(logits, targets) - result["loss"]
                sharp.append(val.item())
                
                # maclaurin
                wx = inputs @ w1.T
                sigmoid_apx = (torch.exp(-wx) + 1) ** -1
                sigmoid_apx += ((torch.exp(-wx) - 1) * torch.exp(-wx)) / (2 * (torch.exp(-wx) + 1) ** 3) * ((sigma * norm) ** 2).view(-1, 1)
                logits_apx = sigmoid_apx @ w2.T
                val = F.cross_entropy(logits_apx, targets) - result["loss"]
                sharp_apx1.append(val.item())
                
                sigmoid_apx += (((torch.exp(-3 * wx) - 11 * torch.exp(-2 * wx) + 11 * torch.exp(-wx) -1) * torch.exp(-wx) )/ (8 * (torch.exp(-wx) + 1) ** 5)) * ((sigma * norm) ** 4).view(-1, 1)
                logits_apx = sigmoid_apx @ w2.T
                val = F.cross_entropy(logits_apx, targets) - result["loss"]
                sharp_apx2.append(val.item())
                
                # probit
                cdf = torch.distributions.normal.Normal(0,1).cdf
                denom = torch.sqrt(torch.tensor(8/pi) + (sigma * norm) ** 2)
                sigmoid_apx = cdf(wx / denom.view(-1, 1))
                logits_apx = sigmoid_apx @ w2.T
                val = F.cross_entropy(logits_apx, targets) - result["loss"]
                sharp_apx3.append(val.item())
            
            result["sharp"] = sharp
            result["sharp_apx1"] = sharp_apx1
            result["sharp_apx2"] = sharp_apx2
            result["sharp_apx3"] = sharp_apx3

        return result
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)
        
        sharp = [result["sharp"] for result in training_step_outputs]
        sharp_apx1 = [result["sharp_apx1"] for result in training_step_outputs]
        sharp_apx2 = [result["sharp_apx2"] for result in training_step_outputs]
        sharp_apx3 = [result["sharp_apx3"] for result in training_step_outputs]
        
        SHARP.extend(sharp)
        SHARP_APX1.extend(sharp_apx1)
        SHARP_APX2.extend(sharp_apx2)
        SHARP_APX3.extend(sharp_apx3)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def experiment(args):
    global SHARP, SHARP_APX1, SHARP_APX2, SHARP_APX3
                      
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
    SHARP_APX3 = np.asarray(SHARP_APX3)

    with open(osp.join(args.out_dir, "sharp.pkl"), "wb") as f:
        pickle.dump(np.concatenate((SHARP, SHARP_APX1, SHARP_APX2, SHARP_APX3), axis=0), f)
    
    for j, sigma in enumerate(SIGMA):
        sharp = moving_average(SHARP[:, j], MOVING_AVG)
        sharp_apx1 = moving_average(SHARP_APX1[:, j], MOVING_AVG)
        sharp_apx2 = moving_average(SHARP_APX2[:, j], MOVING_AVG)
        sharp_apx3 = moving_average(SHARP_APX3[:, j], MOVING_AVG)
        
        x = np.arange(len(sharp))
        plt.plot(x, sharp, label=f"Actual - {MC_SAMPLES} MC samples")
        plt.plot(x, sharp_apx1, label="Maclaurin - 2 terms")
        plt.plot(x, sharp_apx2, label="Maclaurin - 3 terms")
        plt.plot(x, sharp_apx3, label="Probit")
        plt.xlabel("Step (Moving Avg)")
        plt.ylabel("Sharpness")
        plt.legend()
        plt.title(f"MNIST, (w: {args.mlp_hidden_dim}, d: {args.mlp_num_layers}), SGD 0.05, SIGMA {sigma}, B 256, LOSS {LOSS_THRESH}")
        plt.savefig(osp.join(args.out_dir, f"sharpness_sigma{sigma}.png"))
        plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
