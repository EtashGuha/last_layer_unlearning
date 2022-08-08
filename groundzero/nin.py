from torch import nn

from groundzero.model import Model


class NiN(Model):
    def __init__(self, args, classes):
        super().__init__(args, classes)
        
        self.model = nn.Sequential()
        
        h = [args.nin_width for _ in range(args.nin_num_layers)]
        
        channels = zip([args.nin_input_dim] + h[:-1], h)
        for j, (n, k) in enumerate(channels):
            self.model.append(nn.Conv2d(n, k, 3, stride=2, bias=args.bias, padding=args.nin_padding))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.Conv2d(k, k, 1, bias=args.bias, padding=args.nin_padding))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.Conv2d(k, k, 1, bias=args.bias, padding=args.nin_padding))
            self.model.append(nn.ReLU(inplace=True))
            self.model.append(nn.Dropout(args.dropout_prob))
                              
        # Max pooling?
        
        self.model.append(nn.LazyConv2d(args.classes, 1, bias=args.bias, padding=args.nin_padding))
        self.model.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.model.append(nn.Flatten())
    
    def forward(self, inputs):
        outputs = self.model(inputs)
        
        return outputs
