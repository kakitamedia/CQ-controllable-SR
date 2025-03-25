import torch
from torch import nn

from losses import register, make

@register('l1')
class L1Loss(nn.Module):
    def __init__(self, weight=1.0, log_key='l1', **kwargs):
        super().__init__()

        self.fn = nn.L1Loss()
        self.weight = weight
        self.log_key = log_key
        self.pred_key = kwargs['pred_key']
        self.target_key = kwargs['target_key']

    def forward(self, pred, gt):
        loss = self.fn(pred[self.pred_key], gt[self.target_key])
        loss_dict = {self.log_key: loss}
        return loss * self.weight, loss_dict