from testbed.args import parse_args
from testbed.main import main
from testbed.resnet import ResNet

class ResNetExp(ResNet):
    def __init__(args):
        super().__init__(args)

    def validation_epoch_end(self, batch, idx):
        # Plot

    def test_epoch_end(self, batch, idx):
        # Plot

if __name__ == "__main__":
    args = parse_args()
    main(args, model_class=ResNetExp)

