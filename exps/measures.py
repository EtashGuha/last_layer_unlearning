from copy import deepcopy
from math import log, pi
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

SIGMA = [0.1]
TRAIN_ACC = []
TEST_ACC = []
SHARP = []
SHARP_APX1 = []
SHARP_APX2 = []
PROD_SPEC = []
PROD_FRO = []
MARGIN = []


def to_np(x):
    return x.cpu().detach().numpy()

class MeasuresMLP(MLP):
    def __init__(self, args, classes):
        super().__init__(args, classes)
        
        self.fc_layers = [3 * i for i in range(args.mlp_num_layers)]
        
    def training_step(self, batch, idx):
        result = super().training_step(batch, idx)

        with torch.no_grad():
            # Sharpness 2-layer only
            inputs, targets = batch
            inputs = inputs.reshape(inputs.shape[0], -1)
            w1 = deepcopy(self.model[0].weight)
            for sigma in SIGMA:
                avg = []
                for _ in range(10):
                    self.model[0].weight = torch.nn.Parameter(torch.normal(w1, sigma))
                    avg.append(self.model(inputs))
                avg = torch.stack(avg)
                logits = torch.mean(avg, dim=0)
                result["sharp"] = F.cross_entropy(logits, targets)
                
                norm = torch.linalg.vector_norm(inputs, dim=1)
                
                # maclaurin
                wx = inputs @ w1.T
                sigmoid_apx = (torch.exp(-wx) + 1) ** -1
                sigmoid_apx += ((torch.exp(-wx) - 1) * torch.exp(-wx)) / (2 * (torch.exp(-wx) + 1) ** 3) * ((sigma * norm) ** 2).view(-1, 1)
                sigmoid_apx += (((torch.exp(-3 * wx) - 11 * torch.exp(-2 * wx) + 11 * torch.exp(-wx) -1) * torch.exp(-wx) )/ (8 * (torch.exp(-wx) + 1) ** 5)) * ((sigma * norm) ** 4).view(-1, 1)
                logits_apx = sigmoid_apx @ self.model[self.fc_layers[1]].weight.T
                result["sharp_apx1"] = F.cross_entropy(logits_apx, targets)
                
                # probit
                cdf = torch.distributions.normal.Normal(0,1).cdf
                denom = torch.sqrt(torch.tensor(8/pi) + (sigma * norm) ** 2)
                sigmoid_apx = cdf(wx / denom.view(-1, 1))
                logits_apx = sigmoid_apx @ self.model[self.fc_layers[1]].weight.T
                result["sharp_apx2"] = F.cross_entropy(logits_apx, targets)
                
            self.model[0].weight = w1
            
            # Margin
            probs_wo_true = deepcopy(result["probs"])
            inds = torch.repeat_interleave(result["targets"].unsqueeze(1), self.hparams.classes, dim=1)
            probs_wo_true.scatter_(1, inds, 0)
            result["margin"] = torch.gather(result["probs"], 1, inds)[:,0] - torch.max(probs_wo_true, dim=1)[0]

        return result
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)
        
        TRAIN_ACC.append(self.train_acc1)
        
        sharps = to_np(torch.stack([result["sharp"] for result in training_step_outputs]))
        sharps_apx1 = to_np(torch.stack([result["sharp_apx1"] for result in training_step_outputs]))
        sharps_apx2 = to_np(torch.stack([result["sharp_apx2"] for result in training_step_outputs]))
        SHARP.append(sharps.mean())
        SHARP_APX1.append(sharps_apx1.mean())
        SHARP_APX2.append(sharps_apx2.mean())
        
        margins = to_np(torch.cat([result["margin"] for result in training_step_outputs]))
        MARGIN.append(np.percentile(margins, 10))
        
        weights = [self.model[i].weight for i in self.fc_layers]
        PROD_SPEC.append(to_np(torch.stack([torch.linalg.norm(w, ord=2) for w in weights])).prod())
        PROD_FRO.append(to_np(torch.stack([torch.linalg.norm(w, ord="fro") for w in weights])).prod())

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        
        TEST_ACC.append(self.val_acc1)
        
def experiment(args):
    global TRAIN_ACC, TEST_ACC, SHARP, SHARP_APX1, SHARP_APX2, PROD_SPEC, PROD_FRO, MARGIN
                      
    callbacks = [
        EarlyStopping(
            monitor="train_loss",
            stopping_threshold=0.01,
        ),
    ]
    
    args.num_sanity_val_steps = 0
    main(args, MeasuresMLP, callbacks=callbacks)
    
    TRAIN_ACC = np.asarray(TRAIN_ACC)
    TEST_ACC = np.asarray(TEST_ACC)
    SHARP = np.asarray(SHARP)
    SHARP_APX1 = np.asarray(SHARP_APX1)
    SHARP_APX2 = np.asarray(SHARP_APX2)
    PROD_SPEC = np.asarray(PROD_SPEC)
    PROD_FRO = np.asarray(PROD_FRO)
    MARGIN = np.asarray(MARGIN)
                      
    def plot(measure, xlabel, name):
        plt.plot(measure, 1-TRAIN_ACC, label="Train", linestyle="dashed")
        plt.plot(measure, 1-TEST_ACC, label="Test", linestyle="solid")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel("0-1 Error")
        plt.title("MNIST, SGD 0.05, WD 0, B 256, Loss 0.01")
        plt.savefig(osp.join(args.out_dir, f"{name}.png"))
        plt.clf()
        
    x = np.arange(len(SHARP))
    plt.plot(x, SHARP, label="Actual", linestyle="solid")
    plt.plot(x, SHARP_APX1, label="Maclaurin", linestyle="dashed")
    plt.plot(x, SHARP_APX2, label="Probit", linestyle="dotted")
    plt.xlabel("Epoch")
    plt.ylabel("Sharpness")
    plt.legend()
    plt.title("MNIST, SGD 0.05, WD 0, B 256, Loss 0.01")
    plt.savefig(osp.join(args.out_dir, "sharpness.png"))
    plt.clf()
             
    plot(PROD_SPEC, "Product of Spectral Norms", "prod_spec")
    plot(PROD_FRO, "Product of Frobenius Norms", "prod_fro")
    
    # Removes negative margin (on MNIST, the first couple epochs).
    MARGIN = MARGIN[MARGIN>0]
    PROD_SPEC = PROD_SPEC[len(PROD_SPEC)-len(MARGIN):]
    PROD_FRO = PROD_FRO[len(PROD_FRO)-len(MARGIN):]
    TRAIN_ACC = TRAIN_ACC[len(TRAIN_ACC)-len(MARGIN):]
    TEST_ACC = TEST_ACC[len(TEST_ACC)-len(MARGIN):]
    
    plot(PROD_FRO / MARGIN, "Margin-Normalized Product of Spectral Norms", "prod_spec_margin")
    plot(PROD_FRO / MARGIN, "Margin-Normalized Product of Frobenius Norms", "prod_fro_margin")          


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
