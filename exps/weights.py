import os.path as osp

from matplotlib.pyplot import plt

from testbed.args import parse_args
from testbed.main import main
from testbed.resnet import ResNet

class ResNetExp(ResNet):
    def __init__(args):
        super().__init__(args)

    def validation_epoch_end(self, batch, idx):
        if self.current_epoch % 20 == 0:
            plt.hist([param.weight for param in self.model.parameters()], 50)
            plt.xlabel("Weight")
            plt.ylabel("Frequency")
            plt.title(f"ResNet CIFAR-10 Epoch {self.current_epoch} Weight Distribution")
            plt.savefig(osp.join(self.hparams.out_dir, f"weights{self.current_epoch}.png")

if __name__ == "__main__":
    args = parse_args()
    main(args, model_class=ResNetExp)

