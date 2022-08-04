from torch import nn

from groundzero.model import Model


class MLP(Model):
    def __init__(self, args, classes):
        super().__init__(args, classes)
        
        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}
        activation = activations[args.mlp_activation]

        self.model = nn.Sequential()
        
        h = [args.mlp_hidden_dim] * (args.mlp_num_layers - 1)

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
