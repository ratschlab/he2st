import torch
from tqdm import tqdm
import numpy as np


def run_inference_from_dataloader(model, dataloader):
    device = torch.device('cpu')
    model.to(device)  # same device
    model.eval()

    out = []

    with torch.no_grad():
        for patch, coord, _, adj, _, _, _ in tqdm(dataloader):

            # patch = patch.to(device)
            # coord = patch.to(device)
            adj = adj.squeeze(0)
            y, _, _ = model(patch, coord, adj)
            y = y.cpu().detach().numpy()

            out.extend(y)

    return np.array(out).squeeze()
