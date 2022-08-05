from torch import nn

from groundzero.model import Model


class CNN(Model):
    def __init__(self, args, classes):
        super().__init__(args, classes)

        self.model = nn.Sequential()

        h = [args.cnn_initial_width * j for j in [2 ** i for i in range(0, args.cnn_num_layers)]]
        
        channels = zip([args.cnn_input_dim] + h[:-1], h)
        for j, (n, k) in channels:
            self.model.append(nn.Conv2d(n, k, args.cnn_kernel_size, bias=args.bias, padding=args.cnn_padding))
            self.model.append(nn.BatchNorm2d(k))
            self.model.append(nn.ReLU(inplace=True))
            if j != 0:
                self.model.append(nn.MaxPool2d(2))
        
        self.model.append(nn.MaxPool2d(4))
        self.model.append(nn.Flatten())
        self.model.append(nn.LazyLinear(args.classes, bias=args.bias))

        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model[-1].parameters():
                p.requires_grad = True

    def forward(self, inputs):
        outputs = self.model(inputs)

        return outputs
