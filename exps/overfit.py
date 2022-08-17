from copy import deepcopy
import os.path as osp

import torch
from torch.fft import fft2
from torch.linalg import svdvals
from torch.nn import Conv2d, Linear, Parameter
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from groundzero.args import parse_args
from groundzero.datasets.cifar10 import CIFAR10
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.utils import to_np

TRAIN_ACC1 = [0]
MARGIN = [0]
SHARPNESS = [0]
WIDTHS = [10]
NUM_MC_SAMPLES = 10
SIGMA = 0.01


class OverfitCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)
        TRAIN_ACC1[0] = self.train_acc1

        margins = to_np(torch.cat([result["margin"] for result in training_step_outputs]))
        margin = np.percentile(margins, 10)
        MARGIN[0] = margin

        loss = torch.stack([result["loss"] for result in training_step_outputs]).mean().item()
        sharp = np.stack([result["sharp_loss"] for result in training_step_outputs]).mean()
        SHARPNESS[0] = sharp - loss

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        result = super().training_step(batch, batch_idx)

        # TODO: Compute only on final epoch.
        with torch.no_grad():
            probs_wo_true = deepcopy(result["probs"])
            inds = torch.repeat_interleave(result["targets"].unsqueeze(1), self.hparams.num_classes, dim=1)
            probs_wo_true.scatter_(1, inds, 0)
            result["margin"] = torch.gather(result["probs"], 1, inds)[:,0] - torch.max(probs_wo_true, dim=1)[0]

            w = []
            for layer in self.model:
                if isinstance(layer, (Conv2d, Linear)):
                    w.append(layer.weight)

            sharp_loss = []
            for _ in range(NUM_MC_SAMPLES):
                for layer in self.model:
                    if isinstance(layer, (Conv2d, Linear)):
                        layer.weight = Parameter(torch.normal(layer.weight, SIGMA))
                logits = self(inputs)
                sharp_loss.append(F.cross_entropy(logits, targets).cpu())

                counter = 0
                for layer in self.model:
                    if isinstance(layer, (Conv2d, Linear)):
                        layer.weight = w[counter]
                        counter += 1

            result["sharp_loss"] = to_np(sharp_loss).mean()

        return result
    
    def test_epoch_end(self, test_step_outputs):
        super().test_epoch_end(test_step_outputs)
        
        top_svs = []
        for layer in self.model:
            if isinstance(layer, Conv2d):
                transforms = fft2(layer.weight)
                svs = svdvals(transforms)
                top_svs.append(torch.max(svs).item())
            elif isinstance(layer, Linear):
                svs = svdvals(layer.weight)
                top_svs.append(torch.max(svs).item())
        
        prod_spec = np.prod(np.asarray(top_svs))
        self.log("prod_spec", prod_spec)
        
            
def experiment(args):
    global TRAIN_ACC1, MARGIN
    
    train_accs = []
    test_accs = []
    norms = []
    for width in WIDTHS:
        args.cnn_initial_width = width
        result = main(args, OverfitCNN, CIFAR10)
        print(f"Margin: {MARGIN}")
        print(f"Sharpness: {SHARPNESS}")
        train_accs.append(TRAIN_ACC1[0])
        test_accs.append(result[0]["test_acc1"])
        norms.append(result[0]["prod_spec"])

    print(train_accs)
    print(test_accs)
    print(norms)

    errs = 1 - np.asarray(test_accs)
    
    plt.plot(WIDTHS, errs)
    plt.xlabel("CNN Width Parameter")
    plt.ylabel("Test Error")
    plt.title(f"CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.max_epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit_large_real5.png"))
    plt.clf()
    
    errs = [x for _, x in zip(norms, errs)]
    norms = sorted(norms)

    plt.plot(norms, errs)
    plt.xlabel("Product of CNN Spectral Norms")
    plt.ylabel("Test Error")
    plt.title(f"CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.max_epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit_large_real6.png"))
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
