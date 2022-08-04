import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

from groundzero.model import Model


class MLP(Model):
    def __init__(self, args, classes):
        super().__init__(args, classes)

        h = [args.mlp_hidden_dim] * (args.mlp_num_layers - 1)

        if args.mlp_activation == "relu":
            activation = nn.ReLU
        elif args.mlp_activation == "sigmoid":
            activation = nn.Sigmoid

        self.model = nn.Sequential()

        shapes = zip([args.input_dim] + h, h + [args.classes])
        for i, (n, k) in enumerate(shapes):
            if i == args.mlp_num_layers - 1:
                self.model.append(nn.Linear(n, k, bias=args.bias))
            else:
                self.model.append(nn.Linear(n, k, bias=args.bias))
                self.model.append(activation())
                self.model.append(nn.Dropout(args.dropout_prob))

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], -1)
        outputs = self.model(inputs)

        return outputs
