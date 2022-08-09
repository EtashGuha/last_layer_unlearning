import os.path as osp

from torch.fft import fft2
from torch.linalg import svdvals
from torch.nn import Conv2d

import matplotlib.pyplot as plt
import numpy as np

from groundzero.args import parse_args
from groundzero.datasets.cifar10 import CIFAR10
from groundzero.main import main
from groundzero.models.cnn import CNN
from groundzero.utils import to_np

WIDTHS = [2, 4, 6, 8, 10]


class OverfitCNN(CNN):
    def __init__(self, args):
        super().__init__(args)
        
    def test_step(self, batch, batch_idx):
        result = super().test_step(batch, batch_idx)
        
        top_svs = []
        for layer in self.model:
            if isinstance(layer, Conv2d):
                transforms = fft2(layer.weight)
                top_svs.append(svdvals(transforms)[0].item())
        
        result["test_prod_spec"] = np.prod(np.asarray(top_svs))
        self.log("test_prod_spec", result["test_prod_spec"], on_epoch=True, prog_bar=False, sync_dist=True)
        
        return result
            
def experiment(args):
    accs = []
    norms = []
    for width in WIDTHS:
        args.cnn_initial_width = width
        result = main(args, OverfitCNN, CIFAR10)
        accs.append(result[0]["test_acc1"])
        norms.append(result[0]["test_prod_spec"])

    accs = 1 - np.asarray(accs)
    plt.plot(WIDTHS, accs)
    plt.xlabel("CNN Width Parameter")
    plt.ylabel("Test Error")
    plt.legend()
    plt.title(f"CIFAR-10, {args.optimizer} {args.lr}, B {args.batch_size}, {args.max_epochs} epochs")
    plt.savefig(osp.join(args.out_dir, f"overfit.png"))
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
