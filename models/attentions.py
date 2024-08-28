import functools
import math

import torch
from torch import nn, autograd
from torch.nn import functional as F

from models import register, make

register('attention')
class Attention(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim=None, num_heads=1, pos_encoding=None):
        super().__init__()

        if embed_dim is None:
            embed_dim = in_dim

        assert embed_dim % num_heads == 0; "embed_dim should be divisible by num_heads"


        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Conv1d(in_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv1d(in_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

        self.pos_encoding = make({'name': pos_encoding, 'args': {'dim': embed_dim}}) \
            if pos_encoding is not None else None

    def forward(self, x):
        """
        Args:
            x (tensor): [batch, seq_length, channel]

        Returns:
            out (tensor): [batch, seq_length, out_dim]
        """

        q, k, v = self.gen_qkv(x, self.pos_encoding)
        return self.attention(q, k, v)

    def gen_qkv(self, x, pos=None):
        x = x.permute(0, 2, 1) # [batch, channel, seq_length]
        q = self.q(x).permute(0, 2, 1) # [batch, seq_length, embed_dim]
        k = self.k(x).permute(0, 2, 1) # [batch, seq_length, embed_dim]
        v = self.v(x).permute(0, 2, 1) # [batch, seq_length, out_dim]

        if self.pos_encoding is not None:
            q, k = self.pos_encoding(q, k, pos)

        return q, k, v

    def attention(self, q, k, v):
        batch_size, seq_length, _ = q.size()

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)

        qk = q @ k.transpose(2, 3)
        qk = qk / math.sqrt(self.head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_length, head_dim]
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_length, head_dim]
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_length, head_dim]
        qk = qk.view(batch_size, self.num_heads, seq_length, seq_length)  # [batch, num_heads, seq_length, seq_length]

        attn = F.softmax(qk, dim=-1)
        out = attn @ v  # [batch, num_heads, seq_length, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)  # [batch, seq_length, out_dim]

        attn = F.softmax(qk, dim=-1)
        out = attn @ v

        out = out.view(batch_size, seq_length, -1)

        return out

@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length):
    mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool), diagonal=0)
    return mask


@register('linear-attention')
class LinearCausalAttention(Attention):
    def __init__(self, in_dim, out_dim, embed_dim=None, num_heads=1, pos_encoding=None):
        super().__init__(in_dim, out_dim, embed_dim, num_heads, pos_encoding)

        self.feature_fn = lambda x: F.elu(x) + 1

    def attention(self, q, k, v):
        q, k = self.feature_fn(q), self.feature_fn(k)
        mask = _get_causal_mask(q.shape[1]).to(q.device)

        qk = q @ k.transpose(1, 2)
        qk = qk.masked_fill(mask==0, 0)

        denominator = qk.sum(dim=-1) + 1e-10
        numerator = qk @ v

        out = numerator / denominator.unsqueeze(-1)

        return out

    def recurrent(self, x, pos):
        """

        Args:
            x (tensor): [batch, 1, channel]
            pos (int): position of the input sequence
        Returns:
            out (tensor): [batch, 1, out_dim]
        """

        batch_size, seq_length, _ = x.size()

        q, k, v = self.gen_qkv(x, pos=pos)
        q, k = self.feature_fn(q), self.feature_fn(k)

        # q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        # k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        # v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)

        try:
            s = self.S
            z = self.Z
        except:
            s, z = torch.zeros_like(k.transpose(1, 2) @ v), torch.zeros_like(k)

        self.S = s + k.transpose(1, 2) @ v
        self.Z = z + k

        numerator = q @ self.S
        denominator = torch.einsum('blc,blc->bl', q, self.Z) + 1e-10

        out = numerator / denominator.unsqueeze(-1)

        return out

    def flush(self):
        try:
            del self.S
            del self.Z
        except:
            pass

@register('memory-efficient-linear_attention')
class MemoryEfficientLinearCausalAttention(LinearCausalAttention):
    def __init__(self, in_dim, out_dim, embed_dim=None):
        super().__init__(in_dim, out_dim)

        self.numerator = _UnnormalizedLinearCausalAttention.apply

    def attention(self, q, k, v):
        q, k = self.feature_fn(q), self.feature_fn(k)
        denominator = torch.einsum('blc,blc->bl', q, k.cumsum(1)) + 1e-10
        numerator = self.numerator(q, k, v)
        out = numerator / denominator.unsqueeze(-1)

        return out


def _idx(i):
    return (slice(None), slice(i, i + 1, 1), slice(None))

class _UnnormalizedLinearCausalAttention(autograd.Function):
    """
    Computes unnormalized causal attention using only O(N*C) memory.
    https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn/attention.py
    """

    @staticmethod
    def forward(ctx, Q, K, V):
        ctx.save_for_backward(Q, K, V)

        Vnew, S = torch.zeros_like(V), 0
        for i in range(V.shape[1]):
            S = S + K[_idx(i)].transpose(1, 2) @ V[_idx(i)]
            Vnew[_idx(i)] = Q[_idx(i)] @ S
        return Vnew

    @staticmethod
    def backward(ctx, G):
        Q, K, V = ctx.saved_tensors

        dQ, S = torch.zeros_like(Q), 0
        for i in range(V.shape[1]):
            S = S + K[_idx(i)].transpose(1, 2) @ V[_idx(i)]
            dQ[_idx(i)] = G[_idx(i)] @ S.transpose(1, 2)

        dK, dV, S = torch.zeros_like(K), torch.zeros_like(V), 0
        for i in range(V.shape[1] - 1, -1, -1):
            S = S + Q[_idx(i)].transpose(1, 2) @ G[_idx(i)]
            dV[_idx(i)] = K[_idx(i)] @ S
            dK[_idx(i)] = V[_idx(i)] @ S.transpose(1, 2)
        return dQ, dK, dV


# class Attention(nn.Module):
#     def __init__(self, in_dim, out_dim, embed_dim=None):
#         super().__init__()

#         if embed_dim is None:
#             embed_dim = in_dim

#         self.q = nn.Conv1d(in_dim, embed_dim, kernel_size=1, stride=1, padding=0)
#         self.k = nn.Conv1d(in_dim, embed_dim, kernel_size=1, stride=1, padding=0)
#         self.v = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

#     def forward(self, x, mask=None):
#         """
#         Args:
#             x (tensor): [batch, seq_length, channel]

#         Returns:
#             out (tensor): [batch, seq_length, out_dim]
#         """

#         q, k, v = self.gen_qkv(x)
#         qk = torch.einsum('blc,bkc->blk', q, k)

#     def gen_qkv(self, x):
#         x = x.permute(0, 2, 1)
#         q = self.q(x).permute(0, 2, 1)
#         k = self.k(x).permute(0, 2, 1)
#         v = self.v(x).permute(0, 2, 1)

#         return q, k, v

if __name__ == '__main__':
    torch.random.manual_seed(0)
    # x = torch.randn([1, 16, 4])
    x = torch.tensor([i for i in range(16)], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    # model = MemoryEfficientLinearCausalAttention(1, 1)
    model = LinearCausalAttention(1, 1)
    # model = Attention(1, 1)
    # out = model(x)
    out = x[:, 0, :].unsqueeze(1)
    for i in range(x.shape[1]):
        out = torch.cat([out, model.recurrent(x[:, i, :].unsqueeze(1))], dim=1)
    print(x)
    print(out)
    print(x.shape, out.shape)
    loss = out.sum()
    loss.backward()