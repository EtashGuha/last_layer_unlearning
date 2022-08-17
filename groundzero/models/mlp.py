import torch
from torch import nn

from groundzero.models.model import Model


def relu():
    return nn.ReLU(inplace=True)

class MLP(Model):
    def __init__(self, args):
        super().__init__(args)
        
        activations = {"relu": relu, "sigmoid": nn.Sigmoid}
        activation = activations[args.mlp_activation]

        self.model = nn.Sequential()
        
        h = [args.mlp_hidden_dim] * (args.mlp_num_layers - 1)

        shapes = zip([args.mlp_input_dim] + h, h + [args.num_classes])
        for i, (n, k) in enumerate(shapes):
            if i == args.mlp_num_layers - 1:
                self.model.append(nn.Linear(n, k, bias=args.bias))
            else:
                self.model.append(nn.Linear(n, k, bias=args.bias))
                self.model.append(activation())
                self.model.append(nn.Dropout(args.dropout_prob))
                
        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model[-1].parameters():
                p.requires_grad = True

    def load_msg(self):
        return f"Loading MLP with {self.hparams.mlp_num_layers} layers and hidden dimension {self.hparams.mlp_hidden_dim}."

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], -1)
        outputs = torch.squeeze(self.model(inputs), dim=-1)

        return outputs

