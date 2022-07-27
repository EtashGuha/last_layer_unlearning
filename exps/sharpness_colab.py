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

SIGMA = 0.02
MC_SAMPLES = 5
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
            
            avg = []
            for _ in range(MC_SAMPLES):
                avg.append(torch.sigmoid(inputs @ torch.normal(w1, SIGMA).T) @ w2.T)
            logits = torch.mean(torch.stack(avg), dim=0)
            result["sharp"] = F.cross_entropy(logits, targets) - result["loss"]

            # maclaurin
            wx = inputs @ w1.T
            sigmoid_apx = (torch.exp(-wx) + 1) ** -1
            sigmoid_apx += ((torch.exp(-wx) - 1) * torch.exp(-wx)) / (2 * (torch.exp(-wx) + 1) ** 3) * ((SIGMA * norm) ** 2).view(-1, 1)
            sigmoid_apx += (((torch.exp(-3 * wx) - 11 * torch.exp(-2 * wx) + 11 * torch.exp(-wx) -1) * torch.exp(-wx) )/ (8 * (torch.exp(-wx) + 1) ** 5)) * ((SIGMA * norm) ** 4).view(-1, 1)
            logits_apx = sigmoid_apx @ w2.T
            result["sharp_apx1"] = F.cross_entropy(logits_apx, targets) - result["loss"]

            # probit
            cdf = torch.distributions.normal.Normal(0,1).cdf
            denom = torch.sqrt(torch.tensor(8/pi) + (SIGMA * norm) ** 2)
            sigmoid_apx = cdf(wx / denom.view(-1, 1))
            logits_apx = sigmoid_apx @ w2.T
            result["sharp_apx2"] = F.cross_entropy(logits_apx, targets) - result["loss"]

        return result
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)
        
        sharps = to_np(torch.stack([result["sharp"] for result in training_step_outputs]))
        sharps_apx1 = to_np(torch.stack([result["sharp_apx1"] for result in training_step_outputs]))
        sharps_apx2 = to_np(torch.stack([result["sharp_apx2"] for result in training_step_outputs]))
        
        SHARP.append(sharps.mean())
        SHARP_APX1.append(sharps_apx1.mean())
        SHARP_APX2.append(sharps_apx2.mean())
        
def experiment(args):
    global SHARP, SHARP_APX1, SHARP_APX2
                      
    callbacks = [
        EarlyStopping(
            monitor="train_loss",
            stopping_threshold=0.01,
        ),
    ]
    
    args.num_sanity_val_steps = 0
    main(args, SharpnessMLP, callbacks=callbacks)
    
    SHARP = np.asarray(SHARP)
    SHARP_APX1 = np.asarray(SHARP_APX1)
    SHARP_APX2 = np.asarray(SHARP_APX2)
    
    x = np.arange(len(SHARP))
    plt.plot(x, SHARP, label="Actual", linestyle="solid")
    plt.plot(x, SHARP_APX1, label="Maclaurin", linestyle="dashed")
    plt.plot(x, SHARP_APX2, label="Probit", linestyle="dotted")
    plt.xlabel("Epoch")
    plt.ylabel("Sharpness")
    plt.legend()
    plt.title("MNIST, SGD 0.05, SIGMA 0.02, WD 0, B 256, Loss 0.01")
    plt.savefig(osp.join(args.out_dir, "sharpness.png"))
    plt.clf()      


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
