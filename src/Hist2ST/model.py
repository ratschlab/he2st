from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from collections import defaultdict as dfd
from copy import deepcopy as dcp
from torch.utils.data import DataLoader
import torchvision.transforms as tf
import lightning as L
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch.autograd.variable import *
from torch.autograd import Function
from scipy.stats import pearsonr
from torch import nn, einsum
from anndata import AnnData
import scanpy as sc
import pandas as pd
from collections import defaultdict
import random
import time
from torch.autograd import Variable
from torch.nn import init
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


def NB_loss(x, h_r, h_p):

    ll = torch.lgamma(torch.exp(h_r) + x) - torch.lgamma(torch.exp(h_r))
    ll += h_p * x - torch.log(torch.exp(h_p) + 1) * (x + torch.exp(h_r))

    loss = -torch.mean(torch.sum(ll, axis=-1))
    return loss


def ZINB_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
    eps = 1e-10
    if isinstance(scale_factor, float):
        scale_factor = np.full((len(mean),), scale_factor)
    scale_factor = scale_factor[:, None]
    mean = mean * scale_factor

    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0 - pi + eps)
    zero_nb = torch.pow(disp / (disp + mean + eps), disp)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

    if ridge_lambda > 0:
        ridge = ridge_lambda * torch.square(pi)
        result += ridge
    result = torch.mean(result)
    return result


class gs_block(nn.Module):
    def __init__(
        self, feature_dim, embed_dim,
        policy='mean', gcn=False, num_sample=10
    ):
        super().__init__()
        self.gcn = gcn
        self.policy = policy
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.num_sample = num_sample
        self.weight = nn.Parameter(torch.FloatTensor(
            embed_dim,
            self.feat_dim if self.gcn else 2 * self.feat_dim
        ))
        init.xavier_uniform_(self.weight)

    def forward(self, x, Adj):
        neigh_feats = self.aggregate(x, Adj)
        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined, 2, 1)
        return combined

    def aggregate(self, x, Adj):
        adj = Variable(Adj).to(Adj.device)
        if not self.gcn:
            n = len(adj)
            adj = adj - torch.eye(n).to(adj.device)
        if self.policy == 'mean':
            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)
            to_feats = mask.mm(x)
        elif self.policy == 'max':
            indexs = [i.nonzero() for i in adj == 1]
            to_feats = []
            for feat in [x[i.squeeze()] for i in indexs]:
                if len(feat.size()) == 1:
                    to_feats.append(feat.view(1, -1))
                else:
                    to_feats.append(torch.max(feat, 0)[0].view(1, -1))
            to_feats = torch.cat(to_feats, 0)
        return to_feats

############


# from easydl import *


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    # @get_local('attn')
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class attn_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


# from gcn import *
# from NB_module import *
# from transformer import *


class convmixer_block(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = self.dw(x) + x
        x = self.pw(x)
        return x


class mixer_transformer(nn.Module):
    def __init__(self, channel=32, kernel_size=5, dim=1024,
                 depth1=2, depth2=8, depth3=4,
                 heads=8, dim_head=64, mlp_dim=1024, dropout=0.,
                 policy='mean', gcn=True
                 ):
        super().__init__()
        self.layer1 = nn.Sequential(
            *[convmixer_block(channel, kernel_size) for i in range(depth1)],
        )
        self.layer2 = nn.Sequential(*[attn_block(dim, heads, dim_head, mlp_dim, dropout) for i in range(depth2)])
        self.layer3 = nn.ModuleList([gs_block(dim, dim, policy, gcn) for i in range(depth3)])
        self.jknet = nn.Sequential(
            nn.LSTM(dim, dim, 2),
            SelectItem(0),
        )
        self.down = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, 1),
            nn.Flatten(),
        )

    def forward(self, x, ct, adj):
        x = self.down(self.layer1(x))
        g = x.unsqueeze(0)
        g = self.layer2(g + ct).squeeze(0)
        jk = []
        for layer in self.layer3:
            g = layer(g, adj)
            jk.append(g.unsqueeze(0))
        g = torch.cat(jk, 0)
        g = self.jknet(g).mean(0)
        return g


class ViT(nn.Module):
    def __init__(self, channel=32, kernel_size=5, dim=1024,
                 depth1=2, depth2=8, depth3=4,
                 heads=8, mlp_dim=1024, dim_head=64, dropout=0.,
                 policy='mean', gcn=True
                 ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.transformer = mixer_transformer(
            channel, kernel_size, dim,
            depth1, depth2, depth3,
            heads, dim_head, mlp_dim, dropout,
            policy, gcn,
        )

    def forward(self, x, ct, adj):
        x = self.dropout(x)
        x = self.transformer(x, ct, adj)
        return x


##################


class Hist2ST(L.LightningModule):
    def __init__(self, learning_rate=1e-5, fig_size=112, label=None,
                 dropout=0.2, n_pos=64, kernel_size=5, patch_size=7, n_genes=785,
                 depth1=2, depth2=8, depth3=4, heads=16, channel=32,
                 zinb=0.25, nb=True, bake=5, lamb=0.5, policy='mean',
                 ):
        super().__init__()
        # self.save_hyperparameters()
        dim = (fig_size // patch_size)**2 * channel // 8
        self.learning_rate = learning_rate
        self.fig_size = fig_size
        self.nb = nb
        self.zinb = zinb

        self.bake = bake
        self.lamb = lamb

        self.label = label
        self.patch_embedding = nn.Conv2d(3, channel, patch_size, patch_size)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(
            channel=channel, kernel_size=kernel_size, heads=heads,
            dim=dim, depth1=depth1, depth2=depth2, depth3=depth3,
            mlp_dim=dim, dropout=dropout, policy=policy, gcn=True,
        )
        self.channel = channel
        self.patch_size = patch_size
        self.n_genes = n_genes
        if self.zinb > 0:
            if self.nb:
                self.hr = nn.Linear(dim, n_genes)
                self.hp = nn.Linear(dim, n_genes)
            else:
                self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
                self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
                self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())
        if self.bake > 0:
            self.coef = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            )
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )
        self.tf = tf.Compose([
            tf.RandomGrayscale(0.1),
            tf.RandomRotation(90),
            tf.RandomHorizontalFlip(0.2),
        ])

    def forward(self, patches, centers, adj, aug=False):
        B, N, C, H, W = patches.shape
        patches = patches.reshape(B * N, C, H, W)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])
        ct = centers_x + centers_y
        h = self.vit(patches, ct, adj)
        x = self.gene_head(h)
        extra = None
        if self.zinb > 0:
            if self.nb:
                r = self.hr(h)
                p = self.hp(h)
                extra = (r, p)
            else:
                m = self.mean(h)
                d = self.disp(h)
                p = self.pi(h)
                extra = (m, d, p)
        if aug:
            h = self.coef(h)
        return x, extra, h

    def aug(self, patch, center, adj):
        bake_x = []
        for i in range(self.bake):
            new_patch = self.tf(patch.squeeze(0)).unsqueeze(0)
            x, _, h = self(new_patch, center, adj, True)
            bake_x.append((x.unsqueeze(0), h.unsqueeze(0)))
        return bake_x

    def distillation(self, bake_x):
        new_x, coef = zip(*bake_x)
        coef = torch.cat(coef, 0)
        new_x = torch.cat(new_x, 0)
        coef = F.softmax(coef, dim=0)
        new_x = (new_x * coef).sum(0)
        return new_x

    def training_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_ = batch
        adj = adj.squeeze(0)
        exp = exp.squeeze(0)
        pred, extra, h = self(patch, center, adj)

        mse_loss = F.mse_loss(pred, exp)
        self.log('mse_loss', mse_loss, on_epoch=True, prog_bar=True, logger=True)
        bake_loss = 0
        if self.bake > 0:
            bake_x = self.aug(patch, center, adj)
            new_pred = self.distillation(bake_x)
            bake_loss += F.mse_loss(new_pred, pred)
            self.log('bake_loss', bake_loss, on_epoch=True, prog_bar=True, logger=True)
        zinb_loss = 0
        if self.zinb > 0:
            if self.nb:
                r, p = extra
                zinb_loss = NB_loss(oris.squeeze(0), r, p)
            else:
                m, d, p = extra
                zinb_loss = ZINB_loss(oris.squeeze(0), m, d, p, sfs.squeeze(0))
            self.log('zinb_loss', zinb_loss, on_epoch=True, prog_bar=True, logger=True)

        loss = mse_loss + self.zinb * zinb_loss + self.lamb * bake_loss
        self.log('train', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict
