import lightning as L
from torch import nn
import torch


class MLP(L.LightningModule):
    def __init__(self, input_size, output_size, loss_func, middle_layer, p=0, lr=0.0001, weight_decay=0.000001):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.loss_func = loss_func
        self.lr = lr
        self.p = p
        self.middle_layer = middle_layer
        self.weight_decay = weight_decay
        self.training_loss = []
        self.validation_loss = []

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.middle_layer),
            nn.ReLU(True),
            nn.Dropout(p=self.p),
            nn.Linear(self.middle_layer, self.output_size),
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
