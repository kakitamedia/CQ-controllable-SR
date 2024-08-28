import torch
from torch import nn

from models import register, make

@register('sft')
class SFT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.gamma = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.beta = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, cond, *args):
        gamma = self.gamma(cond).expand_as(feat)
        beta = self.beta(cond).expand_as(feat)

        return feat * (1 + gamma) + beta