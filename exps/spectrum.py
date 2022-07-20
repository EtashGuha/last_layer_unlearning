from math import prod
import os.path as osp

import torch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from groundzero.args import parse_args
from groundzero.main import main
from groundzero.mlp import MLP

ACCS = []
NORMS = []
SPECTRA = []
TRAIN_EIGENMAX = []
VAL_EIGENMAX = []

def np(x):
    return x.cpu().detach().numpy()

def singular(x, A):
    # return batched (x^T (A^T A) x / ||x||^2)^(1/2)
    return torch.sqrt(torch.sum(x * torch.matmul(torch.matmul(A.T, A), x.T).T, dim=1) / (torch.linalg.vector_norm(x, dim=1) ** 2))

class SpectrumMLP(MLP):
    def __init__(self, args, classes):
        super().__init__(args, classes)

        self.train_eigenmax = [0] * args.mlp_num_layers
        self.val_eigenmax = [0] * args.mlp_num_layers
        self.fc_layers = [3 * i for i in range(args.mlp_num_layers)]

    def forward(self, inputs):
        x = inputs.reshape(inputs.shape[0], -1)

        for i, layer in enumerate(self.model):
            with torch.no_grad():
                if i in self.fc_layers:
                    j = i // 3
                    v = singular(x, layer.weight)
                    for k in v:
                        if self.training:
                            if k.item() > self.train_eigenmax[j]:
                                self.train_eigenmax[j] = k.item()
                        else:
                            if k.item() > self.val_eigenmax[j]:
                                self.val_eigenmax[j] = k.item()
            x = layer(x)

        return x
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)

        TRAIN_EIGENMAX.append(self.train_eigenmax)
        self.train_eigenmax = [0] * self.hparams.mlp_num_layers

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)

        weights = [self.model[i].weight for i in self.fc_layers]

        ACCS.append([self.train_acc1, self.val_acc1])
        NORMS.append([(np(torch.linalg.norm(w, ord=2)), np(torch.linalg.norm(w, ord="fro"))) for w in weights])
        SPECTRA.append([np(torch.linalg.svdvals(w))[:5] for w in weights])

        VAL_EIGENMAX.append(self.val_eigenmax)
        self.val_eigenmax = [0] * self.hparams.mlp_num_layers

def experiment(args):
    """
    callbacks = [
        EarlyStopping(
            monitor="train_acc1",
            mode="max",
            stopping_threshold=0.99,
            check_on_train_epoch_end=True,
        ),
    ]
    """

    global ACCS, NORMS, SPECTRA, TRAIN_EIGENMAX, VAL_EIGENMAX

    colors = ["red", "blue", "green", "orange", "brown", "purple"]
    params = [(256, 2), (1024, 2), (256, 3), (1024, 3), (256, 4), (1024, 4)]
    legend = [f"(w: {p[0]}, d: {p[1]})" for p in params]

    accs = {}
    norms = {}
    spectra = {}
    train_eigenmax = {}
    val_eigenmax = {}
    for width, depth in params:
        args.mlp_hidden_dim = width
        args.mlp_num_layers = depth
        
        main(args, SpectrumMLP)

        accs[(width, depth)] = ACCS
        norms[(width, depth)] = NORMS
        spectra[(width, depth)] = SPECTRA
        train_eigenmax[(width, depth)] = TRAIN_EIGENMAX
        val_eigenmax[(width, depth)] = VAL_EIGENMAX

        ACCS = []
        NORMS = []
        SPECTRA = []
        TRAIN_EIGENMAX = []
        VAL_EIGENMAX = []

    dashed_line = Line2D([], [], color="black", label="Train", linestyle="dashed")
    solid_line = Line2D([], [], color="black", label="Test", linestyle="solid")
    red_patch = Patch(color="red", label=legend[0])
    blue_patch = Patch(color="blue", label=legend[1])
    green_patch = Patch(color="green", label=legend[2])
    orange_patch = Patch(color="orange", label=legend[3])
    brown_patch = Patch(color="brown", label=legend[4])
    purple_patch = Patch(color="purple", label=legend[5])

    x = [f"(w: {width}, d: {depth})" for width, depth in params]
    train_accs = [accs[(width, depth)][-1][0] for width, depth in params]
    val_accs = [accs[(width, depth)][-1][1] for width, depth in params]

    x_axis = numpy.arange(len(x))
    plt.bar(x_axis - 0.2, train_accs, 0.4, label="Train")
    plt.bar(x_axis + 0.2, val_accs, 0.4, label="Test")
    plt.xticks(x_axis, x, fontsize=8)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim([0.95, 1.0])
    plt.title("MNIST, SGD 0.02/0.002, WD 0, B 256, 80 epochs")
    plt.savefig(osp.join(args.out_dir, "acc.png"))
    plt.clf()
    
    x = list(range(args.max_epochs + 1))
    dashed_line = Line2D([], [], color="black", label="Spectral", linestyle="dashed")
    solid_line = Line2D([], [], color="black", label="Frobenius", linestyle="solid")
    
    spectral = [prod([epoch[w][0] for w in range(depth)]) for epoch in norms[(width, depth)]]
    plt.plot(x, spectral, label=f"(w: {width}, d: {depth})")
    plt.legend(handles=[red_patch, blue_patch, green_patch, orange_patch, brown_patch, purple_patch])
    plt.xlabel("Epoch")
    plt.ylabel("Product of Spectral Norms")
    plt.title("MNIST, SGD 0.02/0.002, WD 0, B 256, 80 epochs")
    plt.savefig(osp.join(args.out_dir, "prods1.png"))
    plt.clf()

    frobenius = [prod([epoch[w][1] for w in range(depth)]) for epoch in norms[(width, depth)]]
    plt.plot(x, frobenius, label=f"(w: {width}, d: {depth})")
    plt.legend(handles=[red_patch, blue_patch, green_patch, orange_patch, brown_patch, purple_patch])
    plt.xlabel("Epoch")
    plt.ylabel("Product of Frobenius Norms")
    plt.title("MNIST, SGD 0.02/0.002, WD 0, B 256, 80 epochs")
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
            plt.legend(handles=[red_patch, green_patch])
        elif depth == 3:
            blue_patch = Patch(color="blue", label="Hidden 2")
            plt.legend(handles=[red_patch, blue_patch, green_patch], loc="center right")
        plt.gca().add_artist(legend1)
        plt.xlabel("Epoch")
        plt.ylabel("Norm")
        plt.title(f"(w: {width}, d: {depth}), MNIST, SGD 0.02/0.002, WD 0, B 256, 80 epochs")
        plt.savefig(osp.join(args.out_dir, f"norms{n}.png"))
        plt.clf()

    plot_weight_norms(1, 256, 2)
    plot_weight_norms(2, 256, 3)
    plot_weight_norms(3, 256, 4)
    plot_weight_norms(4, 1024, 4)

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
        plt.title(f"(w: {width}, d: {depth}), MNIST, SGD 0.02/0.002, WD 0, B 256, 80 epochs")
        plt.savefig(osp.join(args.out_dir, f"spectra{n}.png"))
        plt.clf()

    plot_spectra(1, 256, 2)
    plot_spectra(2, 256, 3)
    plot_spectra(3, 256, 4)
    plot_spectra(4, 1024, 4)

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
        plt.title(f"(w: {width}, d: {depth}), MNIST, SGD 0.02/0.002, WD 0, B 256, 80 epochs")
        plt.savefig(osp.join(args.out_dir, f"eigenmax{n}.png"))
        plt.clf()

    plot_eigenmax(1, 256, 2)
    plot_eigenmax(2, 256, 3)
    plot_eigenmax(3, 256, 4)
    plot_eigenmax(4, 1024, 4)


if __name__ == "__main__":
    args = parse_args()
    experiment(args)

