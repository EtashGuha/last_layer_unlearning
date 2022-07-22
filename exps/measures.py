from copy import deepcopy
from math import log
import os.path as osp

import torch

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from groundzero.args import parse_args
from groundzero.main import main
from groundzero.mlp import MLP
from groundzero.utils import compute_accuracy

TRAIN_ACC = []
TEST_ACC = []
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
            probs_wo_true = deepcopy(result["probs"])
            inds = torch.repeat_interleave(result["targets"].unsqueeze(1), self.hparams.classes, dim=1)
            probs_wo_true.scatter_(1, inds, 0)
            result["margin"] = torch.gather(result["probs"], 1, inds)[:,0] - torch.max(probs_wo_true, dim=1)[0]

        return result
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)
        
        TRAIN_ACC.append(self.train_acc1)
        
        margins = to_np(torch.cat([result["margin"] for result in training_step_outputs]))
        MARGIN.append(np.percentile(margins, 10))

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        
        TEST_ACC.append(self.val_acc1)
                      
        weights = [self.model[i].weight for i in self.fc_layers]
        PROD_SPEC.append(to_np(torch.stack([torch.linalg.norm(w, ord=2) for w in weights])).prod())
        PROD_FRO.append(to_np(torch.stack([torch.linalg.norm(w, ord="fro") for w in weights])).prod())
        
def experiment(args):
    global TRAIN_ACC, TEST_ACC, PROD_SPEC, PROD_FRO, MARGIN
                      
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

    plot(PROD_SPEC, "Product of Spectral Norms", "prod_spec")
    plot(PROD_FRO, "Product of Frobenius Norms", "prod_fro")
    plot(PROD_FRO / MARGIN, "Margin-Normalized Product of Spectral Norms", "prod_spec_margin")
    plot(PROD_FRO / MARGIN, "Margin-Normalized Product of Frobenius Norms", "prod_fro_margin")          


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
