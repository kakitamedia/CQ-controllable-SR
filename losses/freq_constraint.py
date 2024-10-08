import torch
from torch import nn

from losses import register, make

@register('periodicity')
class PeriodicityLoss(nn.Module):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()

        self.fn = nn.L1Loss()
        self.weight = weight
        self.num_pred = kwargs['model'].predictor.num_pred

    def forward(self, pred, gt):
        bs, q = pred['freq'][0].shape[:2]
