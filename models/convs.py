import torch
from torch import nn

class CausalConv1d(nn.Conv1d):
    def __init__(self, mask_center, *args, **kwargs):
        super().__init__(*args, **kwargs)

        i, o, l = self.weight.shape
        mask = torch.zeros((i, o, l))
        mask.data[:, :, :l//2 + int(not mask_center)] = 1
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)