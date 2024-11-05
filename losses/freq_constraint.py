import torch
from torch import nn

from losses import register, make

@register('periodicity')
class PeriodicityLoss(nn.Module):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()

        self.fn = nn.L1Loss()
        self.weight = weight
        self.num_pred = getattr(kwargs['model'].predictor, 'num_pred', 0)
        self.block_size = getattr(kwargs['model'].predictor, 'block_size', 1)

    def forward(self, pred, gt):
        bs, q = pred['freq'][0].shape[:2]
        for i in range(len(pred['coef'])):
            coef, freq = pred['coef'][i].view(bs * q, -1), pred['freq'][i].view(bs * q, -1)

            freq = torch.stack(torch.split(freq, 6*self.block_size, dim=-1), dim=-1)
            repeated = freq[:, ]

# TODO: Implement the following loss function
class CosSimilarityLoss(nn.Module):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()

        self.fn = nn.CosineSimilarity(dim=-1)
        self.weight = weight

    def forward(self, pred, gt):
        return self.fn(pred, gt) * self.weight