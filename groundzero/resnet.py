from torch import nn
import torchvision.models as models

from groundzero.model import Model


class ResNet(Model):
    def __init__(self, args, classes):
        super().__init__(args, classes)

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        self.model = resnets[args.resnet_version](pretrained=args.resnet_pretrained)
        self.model.conv1 = nn.Conv2d(args.cnn_input_dim, 64, kernel_size=7)

        self.model.fc = nn.Sequential(
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(self.model.fc.in_features, classes),
        )

        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.fc.parameters():
                p.requires_grad = True


