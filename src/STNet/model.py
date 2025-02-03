import lightning as L
from torch import nn
import torch


class STNet(L.LightningModule):
    def __init__(self, input_size, output_size, loss_func, lr=1e-5, weight_decay=0, momentum=0.9):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.loss_func = loss_func
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.training_loss = []
        self.validation_loss = []

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
        )

    def loop_step(self, batch, stage):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.log(stage, loss, on_epoch=True, prog_bar=True)
        loss_score = loss.cpu().detach().numpy()

        if stage == "train":
            self.training_loss.append(loss_score)
        elif stage == "val":
            self.validation_loss.append(loss_score)

        return loss

    def training_step(self, batch, batch_idx):
        return self.loop_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.loop_step(batch, "val")

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    momentum=self.momentum)
        return optimizer
