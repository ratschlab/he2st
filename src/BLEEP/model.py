import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F

# import config as CFG
# from modules import ImageEncoder, ProjectionHead, ImageEncoder_ViT, ImageEncoder_ViT_L, ImageEncoder_CLIP, ImageEncoder_resnet101, ImageEncoder_resnet152

import lightning as L


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class BLEEP(L.LightningModule):
    def __init__(
        self,
        image_embedding,
        spot_embedding,
        projection_dim=256,
        temperature=2,
        lr=1e-3,
        weight_decay=1e-3,
        dropout=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding,
                                               projection_dim=projection_dim,
                                               dropout=dropout)  # aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding,
                                              projection_dim=projection_dim,
                                              dropout=dropout)  # 3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = batch["image_features"]
        image_embeddings = self.image_projection(image_features)

        if "reduced_expression" in batch:
            # training
            spot_features = batch["reduced_expression"]
            spot_embeddings = self.spot_projection(spot_features)
            return image_embeddings, spot_embeddings

        else:
            return image_embeddings

    def loop_step(self, batch):
        batch = {k: v.cuda() for k, v in batch.items() if k == "image_features" or k == "reduced_expression"}

        image_embeddings, spot_embeddings = self(batch)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            ((images_similarity + spots_similarity) / 2) / self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)
        loss = loss.mean()
        self.log('loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self.loop_step(batch)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer}


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
