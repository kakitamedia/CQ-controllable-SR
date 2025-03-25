import torch
from torch import nn

from losses import register, make

@register('identity')
class Identity(nn.Module):
    def __init__(self, weight=1.0, log_key='identity', **kwargs):
        super().__init__()

        self.weight = weight
        self.log_key = log_key
        self.pred_key = kwargs['pred_key']

    def forward(self, pred, gt):
        loss = pred[self.pred_key].mean()
        loss_dict = {self.log_key: loss}
        return loss  * self.weight, loss_dict