import math

import torch
from torch import nn
from torch.nn import functional as F

from lpips import LPIPS

from .partial_l1 import partial_reconstruction
from losses import register, make
from utils import normalize

@register('lpips')
class LPIPSLoss(nn.Module):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()

        self.fn = LPIPS(net='alex')
        for param in self.fn.parameters():
            param.requires_grad = False

        self.weight = weight

    def forward(self, pred, gt):
        height = width = int(math.sqrt(pred['recon'].shape[-2]))
        x = pred['recon'].view(-1, 3, height, width)
        y = gt['gt_rgb'].view(-1, 3, height, width)
        loss = self.fn(x, y).mean()
        loss_dict = {'loss': loss.item()}
        return loss * self.weight, loss_dict

@register('partial-lpips')
class PartialLPIPSLoss(LPIPSLoss):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(weight=weight)

        self.model = kwargs['model']
        self.mode = kwargs['mode']
        self.length = kwargs.get('length')
        if self.mode == 'specify':
            assert self.length is not None

        self.num_pred = kwargs['model'].predictor.num_pred

    def forward(self, pred, gt):
        if self.mode == 'specify':
            ret = partial_reconstruction(pred, self.model, self.mode, self.num_pred, self.length)
        elif self.mode == 'random':
            ret = partial_reconstruction(pred, self.model, self.mode, self.num_pred)
        else:
            raise ValueError('Invalid mode. Please specify either "random" or "specify".')

        return super().forward({'recon': ret}, gt)