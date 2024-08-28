import torch
from torch import nn

from models import register, make

@register('liif')
class LIIF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, feat, cell, coord, *args):
        return {
            'pred': torch.cat([feat, cell], dim=-1)
        }