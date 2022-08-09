from torch import nn
import torchvision.models as models

from groundzero.models.model import Model


class ResNet(Model):
    def __init__(self, args):
        super().__init__(args)

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        self.model = resnets[args.resnet_version](pretrained=args.resnet_pretrained)
        self.model.conv1 = nn.Conv2d(args.input_channels, 64, kernel_size=7)

        self.model.fc = nn.Sequential(
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(self.model.fc.in_features, args.num_classes),
        )

        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.fc.parameters():
                p.requires_grad = True

    def load_msg(self):
        if self.hparams.resnet_pretrained:
            return f"Loading ImageNet1K-pretrained ResNet{self.hparams.resnet_version}."
        else:
            return f"Loading ResNet{self.hparams.resnet_version} with no pretraining."

