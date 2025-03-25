import torch
from torch import nn
import einops

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

@register('cos_similarity')
class CosSimilarityLoss(nn.Module):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()

        self.fn = nn.CosineSimilarity(dim=-1)
        self.weight = weight
        self.block_size = getattr(kwargs['model'].predictor, 'block_size', 1)
        assert self.block_size > 1, 'block_size must be greater than 1'
        self.out_dim = kwargs['model'].predictor.out_dim

    def forward(self, pred, gt):
        bs, q = pred['freq'][0].shape[:2]
        total_loss = 0
        for i in range(len(pred['freq'])):
            freq = torch.stack(torch.split(pred['freq'][i], 6*self.block_size, dim=-1), dim=-1)
            freq = einops.rearrange(freq, '... (l xy) t -> ... l xy t', xy=2)
            freq = torch.stack(torch.split(freq, 3, dim=-3), dim=-1).permute(0, 1, 2, 4, 3, 5)
            freq = freq / freq.norm(dim=-2, keepdim=True)
            loss = torch.matmul(freq.transpose(-2, -1), freq)
            loss = torch.abs(loss).mean()
            total_loss += loss
        total_loss = total_loss / len(pred['freq']) 
        loss_dict = {'loss': total_loss.item()}
        return -total_loss * self.weight, loss_dict