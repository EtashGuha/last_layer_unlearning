from copy import deepcopy
from math import log
import os.path as osp

import torch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from groundzero.args import parse_args
from groundzero.main import main
from groundzero.mlp import MLP

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
        result = self.step(batch, idx)
        
        probs_wo_true = deepcopy(result["probs"])
        inds = torch.repeat_interleave(targets.unsqueeze(1), self.hparams.classes, dim=1)
        probs_wo_true.scatter_(1, inds, 0)
        result["margin"] = torch.gather(probs, 1, inds)[:,0] - torch.max(probs_wo_true, dim=1)[0]

        acc1, acc5 = compute_accuracy(result["probs"], result["targets"])
        result["acc1"] = acc1
        self.log("train_acc1", acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc5", acc5, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return result
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)
        
        TRAIN_ACC.append(to_np(self.train_acc1))
        
        margins = to_np(torch.cat([result["margin"] for result in training_step_outputs]))
        MARGIN.append(np.percentile(margins, 10)

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        
        TEST_ACC.append(to_np(self.test_acc1))
                      
        weights = [self.model[i].weight for i in self.fc_layers]
        PROD_SPEC.append(to_np([torch.linalg.norm(w, ord=2)) for w in weights]).prod())
        PROD_FRO.append(to_np([torch.linalg.norm(w, ord="fro")) for w in weights]).prod())
        
def experiment(args):
    global TRAIN_ACC, TEST_ACC, PROD_SPEC, PROD_FRO, MARGIN

    dashed_line = Line2D([], [], color="black", label="Train", linestyle="dashed")
    solid_line = Line2D([], [], color="black", label="Test", linestyle="solid")
    red_patch = Patch(color="red", label=legend[0])
    blue_patch = Patch(color="blue", label=legend[1])
    green_patch = Patch(color="green", label=legend[2])
    orange_patch = Patch(color="orange", label=legend[3])

    x = [f"(w: {width}, d: {depth})" for width, depth in params]
    train_accs = [accs[(width, depth)][-1][0] for width, depth in params]
    val_accs = [accs[(width, depth)][-1][1] for width, depth in params]

    x_axis = numpy.arange(len(x))
    plt.bar(x_axis - 0.2, train_accs, 0.4, label="Train")
    plt.bar(x_axis + 0.2, val_accs, 0.4, label="Test")
    plt.xticks(x_axis, x, fontsize=10)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim([0.95, 1.0])
    plt.title("MNIST, SGD 0.05, WD 0, B 256, 50 epochs")
    plt.savefig(osp.join(args.out_dir, "acc.png"))
    plt.clf()
    
    x = list(range(args.max_epochs + 1))
    dashed_line = Line2D([], [], color="black", label="Spectral", linestyle="dashed")
    solid_line = Line2D([], [], color="black", label="Frobenius", linestyle="solid")
    
    for width, depth in params:
        spectral = [prod([epoch[w][0] for w in range(depth)]) for epoch in norms[(width, depth)]]
        plt.plot(x, spectral, label=f"(w: {width}, d: {depth})")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Product of Spectral Norms")
    plt.title("MNIST, SGD 0.05, WD 0, B 256, 50 epochs")
    plt.savefig(osp.join(args.out_dir, "prods1.png"))
    plt.clf()

    for width, depth in params:
        frobenius = [prod([epoch[w][1] for w in range(depth)]) for epoch in norms[(width, depth)]]
        plt.plot(x, frobenius, label=f"(w: {width}, d: {depth})")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Product of Frobenius Norms")
    plt.title("MNIST, SGD 0.05, WD 0, B 256, 50 epochs")
    plt.savefig(osp.join(args.out_dir, "prods2.png"))
    plt.clf()

    def plot_weight_norms(n, width, depth):
        dashed_line = Line2D([], [], color="black", label="Spectral", linestyle="dashed")
        solid_line = Line2D([], [], color="black", label="Frobenius", linestyle="solid")

        if depth == 2:
            colors = ["red", "green"]
        elif depth == 3:
            colors = ["red", "blue", "green"]

        for c, w in zip(colors, list(range(depth))):
            spectral = [epoch[w][0] for epoch in norms[(width, depth)]]
            frobenius = [epoch[w][1] for epoch in norms[(width, depth)]]
            plt.plot(x, spectral, color=c, linestyle="dashed")
            plt.plot(x, frobenius, color=c, linestyle="solid")

        red_patch = Patch(color="red", label="Hidden 1")
        green_patch = Patch(color="green", label="Output")
        legend1 = plt.legend(handles=[dashed_line, solid_line], loc="center right")
        if depth == 2:
            plt.legend(handles=[red_patch, green_patch], loc="upper left")
        elif depth == 3:
            blue_patch = Patch(color="blue", label="Hidden 2")
            plt.legend(handles=[red_patch, blue_patch, green_patch], loc="upper left")
        plt.gca().add_artist(legend1)
        plt.xlabel("Epoch")
        plt.ylabel("Norm")
        plt.title(f"(w: {width}, d: {depth}), MNIST, SGD 0.05, WD 0, B 256, 50 epochs")
        plt.savefig(osp.join(args.out_dir, f"norms{n}.png"))
        plt.clf()

    plot_weight_norms(1, 256, 2)
    plot_weight_norms(2, 512, 2)
    plot_weight_norms(3, 256, 3)
    plot_weight_norms(4, 512, 3)

    def plot_spectra(n, width, depth):
        if depth == 2:
            colors = ["red", "green"]
        elif depth == 3:
            colors = ["red", "blue", "green"]

        for c, w in zip(colors, list(range(depth))):
            spectrum = [epoch[w] for epoch in spectra[(width, depth)]]
            for val_num in range(len(spectrum[0])):
                s = [epoch[val_num] for epoch in spectrum]
                plt.plot(x, s, color=c)

        red_patch = Patch(color="red", label="Hidden 1")
        green_patch = Patch(color="green", label="Output")
        if depth == 2:
            plt.legend(handles=[red_patch, green_patch])
        elif depth == 3:
            blue_patch = Patch(color="blue", label="Hidden 2")
            plt.legend(handles=[red_patch, blue_patch, green_patch])
        plt.xlabel("Epoch")
        plt.ylabel("Singular Values")
        plt.title(f"(w: {width}, d: {depth}), MNIST, SGD 0.05, WD 0, B 256, 50 epochs")
        plt.savefig(osp.join(args.out_dir, f"spectra{n}.png"))
        plt.clf()

    plot_spectra(1, 256, 2)
    plot_spectra(2, 512, 2)
    plot_spectra(3, 256, 3)
    plot_spectra(4, 512, 3)

    def plot_eigenmax(n, width, depth):
        dashed_line = Line2D([], [], color="black", label="max train (xTATAx / ||x||^2)^1/2", linestyle="dashed")
        dotted_line = Line2D([], [], color="black", label="max test (xTATAx / ||x||^2)^1/2", linestyle="dotted")
        solid_line = Line2D([], [], color="black", label="Top SV", linestyle="solid")

        if depth == 2:
            colors = ["red", "green"]
        elif depth == 3:
            colors = ["red", "blue", "green"]

        for c, w in zip(colors, list(range(depth))):
            train = [epoch[w] for epoch in train_eigenmax[(width, depth)]]
            test = [epoch[w] for epoch in val_eigenmax[(width, depth)]]
            spectral = [epoch[w][0] for epoch in norms[(width, depth)]]

            plt.plot(x[1:], train, color=c, linestyle="dashed")
            plt.plot(x, test, color=c, linestyle="dotted")
            plt.plot(x, spectral, color=c, linestyle="solid")

        red_patch = Patch(color="red", label="Hidden 1")
        green_patch = Patch(color="green", label="Output")
        legend1 = plt.legend(handles=[dashed_line, dotted_line, solid_line])
        if depth == 2:
            plt.legend(handles=[red_patch, green_patch], loc="upper left")
        elif depth == 3:
            blue_patch = Patch(color="blue", label="Hidden 2")
            plt.legend(handles=[red_patch, blue_patch, green_patch], loc="upper left")
        plt.gca().add_artist(legend1)
        plt.xlabel("Epoch")
        plt.ylabel("Singular Values")
        plt.title(f"(w: {width}, d: {depth}), MNIST, SGD 0.05, WD 0, B 256, 50 epochs")
        plt.savefig(osp.join(args.out_dir, f"eigenmax{n}.png"))
        plt.clf()

    plot_eigenmax(1, 256, 2)
    plot_eigenmax(2, 512, 2)
    plot_eigenmax(3, 256, 3)
    plot_eigenmax(4, 512, 3)


if __name__ == "__main__":
    args = parse_args()
    experiment(args)
