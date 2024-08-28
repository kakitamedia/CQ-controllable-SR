import torch
import torch.nn as nn

from models import register, make

from .attentions import *
from .convs import CausalConv1d

@register('transformer')
class Transformer(nn.Module):

    def __init__(self, in_dim, embed_dim, out_dim, num_layers=4, pos_encoding=None):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerBlock(4, 1))
        self.layers = nn.ModuleList(layers)

        self.input_layer = nn.Conv1d(1, 4, kernel_size=1, stride=1, padding=0)
        self.out_layer = nn.Conv1d(4, 1, kernel_size=1, stride=1, padding=0)
        self.output_token = nn.Parameter(torch.zeros([out_dim, 4]))

    def forward(self, x):
        x = self.input_layer(x.unsqueeze(1)).permute(0, 2, 1)
        token = self.output_token.unsqueeze(0).expand(x.shape[0], -1, -1)
        for layer in self.layers:
            x, token = layer(x, token=token)
        x = self.out_layer(token.permute(0, 2, 1))
        return x.squeeze(1)


class TransformerBlock(nn.Module):
    def __init__(self, channel, num_heads=1, pos_encoding=None):
        super(TransformerBlock, self).__init__()

        self.attn = Attention(channel, channel, num_heads=num_heads, pos_encoding=pos_encoding)
        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv1d(channel, channel, kernel_size=1, stride=1, padding=0),
        )
        self.norm1 = nn.LayerNorm(channel)
        self.norm2 = nn.LayerNorm(channel)

    def forward(self, x, token=None):
        """
        Args:
            x (tensor): [batch, seq_length, channel]
            token (tensor): additional token (e.g., cls token) [batch, num_token, channel]
        Returns:
            x (tensor): [batch, seq_length, channel]
            token (tensor): [batch, num_token, channel]
        """

        if token is not None:
            token_len = token.shape[1]
            x = torch.cat([token, x], dim=1)

        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = x.permute(0, 2, 1) # [batch, channel, seq_length]
        x = self.conv(x)
        x = x.permute(0, 2, 1) # [batch, seq_length, channel]
        x = x + residual

        if token is not None:
            token, x = x[:, :token_len], x[:, token_len:]
            return x, token

        return x

@register('recurrent-transformer')
class RecurrentTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim, num_layers=4, memory_efficient=False, pos_encoding=None, n_condition=None):
        super().__init__()

        self.in_dim = in_dim
        self.input_layer = nn.Conv1d(in_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.causal_conv = CausalConv1d(True, embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        layers = []
        for _ in range(num_layers):
            layers.append(CausalTransformerBlock(embed_dim, memory_efficient=memory_efficient, pos_encoding=pos_encoding))
        self.layers = nn.ModuleList(layers)

        self.out_layer = nn.Conv1d(embed_dim, out_dim, kernel_size=1, stride=1, padding=0)

        self.n_condition = n_condition
        self.n_condition_layer = make({'name': 'sft', 'args': {'in_dim': 1, 'out_dim': embed_dim}}) if n_condition else None
        if self.n_condition is not None:
            for param in self.n_condition_layer.parameters():
                nn.init.normal_(param, std=0.01)


    def forward(self, x, token=None):
        """
        Args:
            x (tensor) : [batch, channel, seq_length]
            token (tensor) : [batch, channel, num_token]
        """

        if self.n_condition is not None:
            num_pred = x.shape[-1]
            cond = torch.tensor([num_pred], device=token.device, dtype=token.dtype).expand(x.shape[0], 1, 1)

        if num_pred > 0:
            latent = self.input_layer(x)
        latent = latent.permute(0, 2, 1) # [batch, seq_length, channel]

        if self.n_condition == 'step':
            latent = self.n_condition_layer(latent, cond)

        if token is not None:
            token_ = token.permute(0, 2, 1)
            token_len = token.shape[1]
            if self.n_condition == 'token':
                token_ = self.n_condition_layer(token_, cond)
            x = torch.cat([token_, latent], dim=1)

        latent = self.causal_conv(latent)
        for layer in self.layers:
            latent = layer(latent)

        if token is not None:
            latent = latent[:, token_len:, :]

        latent = latent.permute(0, 2, 1) # [batch, channel, seq_length]

        x = self.out_layer(latent)

        return x

    def recurrent(self, x=None, token=None, num_pred=128):
        """
        Args:
            x (tensor) : [batch, channel, seq_length]
            token (tensor) : [batch, channel, num_token]
            num_samples (int) : number of samples to generate
        """

        assert not ((x is None) and (token is None)), 'x and token cannot be None at the same time'
        if x is None:
            x = torch.zeros([token.shape[0], self.in_dim, 0], dtype=token.dtype, device=token.device)
            latent = torch.zeros([token.shape[0], token.shape[1], 0], dtype=token.dtype, device=token.device)

        if self.n_condition is not None:
            cond = torch.tensor([num_pred], device=token.device, dtype=token.dtype).expand(x.shape[0], 1, 1)

        # flush the cache
        for layer in self.layers:
            layer.flush()

        for i in range(x.shape[-1], num_pred):
            if x.shape[-1] > 0:
                latent = x[:, :, i-1:i] # [batch, channel, 1]
                latent = self.input_layer(latent)
                if self.n_condition == 'step':
                    latent = self.n_condition_layer(latent, cond)

            elif token is not None:
                token_ = token
                if self.n_condition == 'token':
                    token_ = self.n_condition_layer(token_, cond)
                latent = torch.cat([token_, latent], dim=-1)

            latent = self.causal_conv(F.pad(latent, (0, 1))) # [batch, channel, 2]

            latent = latent.permute(0, 2, 1) # [batch, 2, channel]

            latent = latent[:, -1, :].unsqueeze(-2) # [batch, 1, channel]

            for layer in self.layers:
                latent = layer.recurrent(latent, pos=i)

            latent = latent.permute(0, 2, 1) # [batch, channel, 1]

            latent = self.out_layer(latent)
            x = torch.cat([x, latent], dim=-1)

        # MEMO: for debugging
        if num_pred == 0:
            x[:, :, :] = 0

        return x[:, :, -num_pred:]

class CausalTransformerBlock(TransformerBlock):
    def __init__(self, channel, memory_efficient=False, pos_encoding=None):
        super().__init__(channel, pos_encoding=pos_encoding)

        attn = MemoryEfficientLinearCausalAttention if memory_efficient else LinearCausalAttention

        self.attn = attn(channel, channel, pos_encoding=pos_encoding)

    def recurrent(self, x, pos=None):
        """
        Args:
            x (tensor): [batch, 1, channel]
        Returns:
            x (tensor): [batch, 1, channel]
        """

        residual = x
        x = self.norm1(x)
        x = self.attn.recurrent(x, pos=pos)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = x.permute(0, 2, 1) # [batch, channel, seq_length]
        x = self.conv(x)
        x = x.permute(0, 2, 1) # [batch, seq_length, channel]
        x = x + residual

        return x

    def flush(self):
        self.attn.flush()



# class RecurrentTransformerBlock(nn.Module):
#     def __init__(self, embed_dim, self_attention, cross_attention=None, dropout=0.1):
#         super().__init__()

#         self.self_attention = self_attention
#         self.cross_attention = cross_attention

#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.norm3 = nn.LayerNorm(embed_dim)

#         self.linear1 = nn.Linear(embed_dim, embed_dim)
#         self.linear2 = nn.Linear(embed_dim, embed_dim)

#         self.dropout = nn.Dropout(dropout)

#         self.activation = nn.GELU()

#     def forward(self, x, memory, mask=None, state=None):
#         """
#         Args:
#             x (tensor): [batch, embed_dim]
#             memory (tensor): [batch, seq_length, embed_dim]
#             mask (tensor): [batch, seq_length, seq_length] ?
#         Returns:
#             x (tensor): [batch, embed_dim]
#         """

#         self_state, cross_state = state or [None, None]

#         # self attention
#         residual = x
#         x, self_state = self.self_attention(x, x, x, state=self_state)
#         x = self.norm1(residual + self.dropout(x))

#         # cross attention
#         residual = x
#         x, cross_state = self.cross_attention(x, memory, memory, mask=mask, state=cross_state)
#         x = self.norm2(residual + self.dropout(x))

#         # fc layer
#         residual = x
#         x = self.dropout(self.activation(self.linear1(x)))
#         x = self.dropout(self.linear2(x))
#         x = self.norm3(residual + x)

#         return x, [self_state, cross_state]


# class RecurrentTransformer(nn.Module):
#     def __init__(self, layers, norm_layer=None):
#         super().__init__()

#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer

#     def forward(self, x, memory, mask=None, state=None):
#         """
#         Args:
#             x (tensor): [batch, seq_length, embed_dim]
#             memory (tensor): [batch, seq_length, embed_dim]
#             mask (tensor): [batch, seq_length, seq_length] ?
#         Returns:
#             x (tensor): [batch, seq_length, embed_dim]
#         """

#         if state is None:
#             state = [None] * len(self.layers)

#         for i, layer in enumerate(self.layers):
#             x, s = layer(x, memory, mask=mask, state=state[i])
#             state[i] = s

#         if self.norm is not None:
#             x = self.norm(x)

#         return x, state


# class TransformerBlock(nn.Module):
    


if __name__ == '__main__':
    x = torch.rand([2, 100, 4])

    model = RecurrentTransformer()
    y = model(x)
    print(y.shape)