from torch import nn

from groundzero.models.model import Model


class NiN(Model):
    def __init__(self, args):
        super().__init__(args)
        
        self.model = nn.Sequential()
        
        h = [args.nin_width for _ in range(args.nin_num_layers)]
        
        channels = zip([args.input_channels] + h[:-1], h)
        for j, (n, k) in enumerate(channels):
            self.model.append(nn.Conv2d(n, k, 3, stride=2, bias=args.bias, padding=args.nin_padding))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.Conv2d(k, k, 1, bias=args.bias, padding=args.nin_padding))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.Conv2d(k, k, 1, bias=args.bias, padding=args.nin_padding))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.MaxPool2d(3, stride=2))
            self.model.append(nn.Dropout(args.dropout_prob))
        
        conv = nn.Conv2d(k, args.num_classes, 1, bias=args.bias, padding=args.nin_padding)
        self.model.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.model.append(nn.Flatten())

    def load_msg(self):
        return f"Loading NiN with {self.hparams.nin_num_layers} layers and width {self.hparams.nin_width}."
    
    def forward(self, inputs):
        outputs = self.model(inputs)
        
        return outputs

