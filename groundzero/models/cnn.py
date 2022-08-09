from torch import nn

from groundzero.models.model import Model


class CNN(Model):
    def __init__(self, args):
        super().__init__(args)
        
        self.model = nn.Sequential()
        
        h = [args.cnn_initial_width * j for j in [2 ** i for i in range(args.cnn_num_layers - 1)]]
        
        channels = zip([args.input_channels] + h[:-1], h)
        for j, (n, k) in enumerate(channels):
            self.model.append(nn.Conv2d(n, k, args.cnn_kernel_size, bias=args.bias, padding=args.cnn_padding))
            if args.cnn_batchnorm:
                self.model.append(nn.BatchNorm2d(k))
            self.model.append(nn.ReLU(inplace=True))
            
            if j != 0:
                self.model.append(nn.MaxPool2d(2))
        
        self.model.append(nn.MaxPool2d(4))
        self.model.append(nn.Flatten())
        self.model.append(nn.LazyLinear(args.num_classes, bias=args.bias))

        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model[-1].parameters():
                p.requires_grad = True

    def load_msg(self):
        return f"Loading CNN with {self.hparams.cnn_num_layers} layers and initial width {self.hparams.cnn_initial_width}."

    def forward(self, inputs):
        outputs = self.model(inputs)
        
        return outputs

