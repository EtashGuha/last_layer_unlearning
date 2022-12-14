"""BERT model implementation."""

# Imports Python builtins.
import types

# Imports Python packages.
from transformers import BertForSequenceClassification, get_scheduler

# Imports PyTorch packages.
from torch.optim import AdamW

# Imports groundzero packages.
from groundzero.models.model import Model


class BERT(Model):
    """BERT model implementation."""

    def __init__(self, args):
        """Initializes a BERT model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__(args)

        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_classes)

        self.model.fc = self.model.classifier
        delattr(self.model, "classifier")

        def classifier(self, x):
            return self.fc(x)
        
        self.model.classifier = types.MethodType(classifier, self.model)
        self.model.base_forward = self.model.forward

        def forward(self, x):
            return self.base_forward(
                input_ids=x[:, :, 0],
                attention_mask=x[:, :, 1],
                token_type_ids=x[:, :, 2]).logits

        self.model.forward = types.MethodType(forward, self.model)

        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.fc.parameters():
                p.requires_grad = True

    def load_msg(self):
        return f"Loading BERT Base Uncased pretrained on Book Corpus and English Wikipedia."

    def configure_optimizers(self):
        if self.hparams.optimizer != "adamw":
            raise NotImplementedError

        no_decay = ["bias", "LayerNorm.weight"]
        decay_params = []
        nodecay_params = []
        for n, p in self.model.named_parameters():
            if not any(nd in n for nd in no_decay):
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            eps=1e-8,
        )

        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.hparams.max_epochs, # should be steps?
        )

        return optimizer

