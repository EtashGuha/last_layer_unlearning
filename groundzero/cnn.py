from torch import nn

from groundzero.model import Model


class CNN(Model):
    def __init__(self, args, classes):
        super().__init__(args, classes)

        self.model = nn.Sequential()

        h = args.cnn_initial_width * [2 ** i for i in range(0, args.cnn_num_layers - 1)]
        
        channels = zip([args.input_dim] + h[:-1], h)
        for n, k in channels:
          self.model.append(nn.Conv2d(n, k, args.cnn_kernel_size, bias=args.bias))
          self.model.append(nn.BatchNorm2d(k))
          self.model.append(nn.ReLU())
          self.model.append(nn.MaxPool2d(2))
        
        self.model.append(nn.Flatten())
        self.model.append(nn.Linear(h[-1], args.classes, bias=args.bias))
        
        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model[-1].parameters():
                p.requires_grad = True

    def forward(self, inputs):
        outputs = self.model(inputs)

        return outputs
