import torch
from torch import nn

from models import register

@register('identity')
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args):
        return {'pred': x}