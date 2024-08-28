import torch
import torch.nn as nn

from models import register


@register('accumulator')
class Accumulator(nn.Module):
    def __init__(self, out_dim, scaling=False):
        super(Accumulator, self).__init__()

        self.out_dim = out_dim
        self.scaling = scaling

    def forward(self, x, scale=1):
        pred = x['decoded']
        batch, channel = pred.shape
        assert channel % self.out_dim == 0, f'channel={channel}, out_dim={self.out_dim}'

        pred = torch.stack(torch.split(pred, self.out_dim, dim=-1), dim=-1)
        pred = pred.sum(dim=-1)
        if self.scaling:
            pred = pred / scale

        x['pred'] = pred

        return x

@register('channeled-accumulator')
class ChanneledAccumulator(nn.Module):
    def __init__(self, out_dim):
        super(ChanneledAccumulator, self).__init__()

        self.out_dim = out_dim

    def forward(self, x, scale=1):
        pred, class_id = x['decoded'], x['class_id']
        batch, channel = pred.shape

        pred = pred[:, :channel//2] + pred[:, channel//2:]

        out = torch.zeros([pred.shape[0], self.out_dim], dtype=pred.dtype, device=pred.device)
        out.scatter_add_(1, class_id, pred)
        out = out / scale

        x['pred'] = out

        return x

@register('non-linear-accumulator')
class NonlinearAccumulator(nn.Module):
    def __init__(self, out_dim, in_dim=64):
        super(NonlinearAccumulator, self).__init__()

        self.layer = nn.Linear(in_dim+2, out_dim)

    def forward(self, x):
        pred = x['pred']
        x['pred'] = self.layer(pred)
        return x

if __name__ == '__main__':
    x = torch.tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12]], dtype=torch.float32)
    id = torch.tensor([0, 1, 0, 0])
    out = torch.zeros_like(x)
    out.scatter_add_(0, id.unsqueeze(1).expand_as(x), x)
    print(out)
    print(x.shape)
    print(id.shape)