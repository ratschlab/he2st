import torch
from tqdm import tqdm
import numpy as np


def calcADJ(coord, k=4, distanceType='euclidean', pruneTag='Grid'):
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    spatialMatrix = coord  # .cpu().numpy()
    nodes = spatialMatrix.shape[0]
    Adj = torch.zeros((nodes, nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp = spatialMatrix[i, :].reshape(1, -1)
        distMat = distance.cdist(tmp, spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0] - 1
        res = distMat.argsort()[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)  # optional
        for j in np.arange(1, k + 1):
            # No prune
            if pruneTag == 'NA':
                Adj[i][res[0][j]] = 1.0
            elif pruneTag == 'STD':
                if distMat[0, res[0][j]] <= boundary:
                    Adj[i][res[0][j]] = 1.0
            # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
            elif pruneTag == 'Grid':
                if distMat[0, res[0][j]] <= 2.0:
                    Adj[i][res[0][j]] = 1.0
    return Adj


def run_inference_from_dataloader(model, dataloader):
    device = torch.device('cpu')
    model.to(device)  # same device
    model.eval()

    out = []

    with torch.no_grad():
        for patch, coord, _, adj in tqdm(dataloader):

            # patch = patch.to(device)
            # coord = patch.to(device)
            y = model(patch, coord, adj)
            y = y.cpu().detach().numpy()

            out.extend(y)

    return np.concatenate(out).squeeze()
