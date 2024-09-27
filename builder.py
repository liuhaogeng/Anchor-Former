import re
import torch
import torch.nn as nn

from torch import einsum
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch.utils.checkpoint import checkpoint
from functools import partial
from timm.models.layers import LayerNorm2d
from timm.models.regnet import RegStage



def exists(val):
    return val is not None


def FeedForward(
    dim,
    mult=4,
    enable_init_network_params=False,
    initializer_range=0.02,
):
    inner_dim = int(dim * mult)
    net = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

    return net


class CAbstractor(nn.Module):
    def __init__(
        self,
        dim,
        depth=3,
        lenth_tokens=144,
    ):
        super().__init__()

        assert (lenth_tokens ** 0.5).is_integer(), "lenth_tokens must be square number"
        hw = int(lenth_tokens ** 0.5)


        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(depth, dim, dim)
        
        sampler=nn.AdaptiveAvgPool2d((hw, hw))
        
        s2 = RegBlock(depth, dim, dim)

        self.net = nn.Sequential(s1, sampler, s2)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, v, D)
        Returns:
            shape (b, n, D) where n is sqrt(v)/2)
        """

        x = x[:, 1:, :].clone()
        assert (x.shape[1] ** 0.5).is_integer(), "image_tokens must be square number"
        hw = int(x.shape[1] ** 0.5)
        
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
    
        return x


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        enable_init_network_params=False,
        initializer_range=0.02,
    ):
        super().__init__()

        self.scale = dim_head**-0.5
        self.heads = heads
        self.initializer_range = initializer_range
        self.use_ft_flash_attention = False

        inner_dim = dim_head * heads


        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        if enable_init_network_params:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents.contiguous())

        h = self.heads

        q = self.to_q(latents)
        # kv_input = torch.cat((x, latents), dim=-2)
        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale
        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=144,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
        enable_init_network_params=False,
        initializer_range=0.02,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing
        self.initializer_range = initializer_range

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            enable_init_network_params=enable_init_network_params,
                            initializer_range=initializer_range,
                        ),
                        FeedForward(
                            dim=dim,
                            mult=ff_mult,
                            enable_init_network_params=enable_init_network_params,
                            initializer_range=initializer_range,
                        ),
                    ]
                )
            )
        # Should this norm layer also change?
        self.norm = nn.LayerNorm(dim)
        if enable_init_network_params:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.initializer_range)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        # b = x.shape[0]
        # latents = repeat(self.latents, "n d -> b n d", b=b)
        latents = self.dynamic_query
        # return latents
        for attn, ff in self.layers:
            if self.gradient_checkpointing and latents.requires_grad:
                latents = checkpoint(attn, x, (latents)) + latents
                latents = checkpoint(ff, latents) + latents
            else:
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents

        return self.norm(latents)




class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    # if mlp_gelu_match:
    #     mlp_depth = int(mlp_gelu_match.group(1))
    #     modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    #     for _ in range(1, mlp_depth):
    #         modules.append(nn.GELU())
    #         modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    #     return nn.Sequential(*modules)
    if mlp_gelu_match:
        perceiver = PerceiverResampler(
            dim=1024,
            enable_init_network_params=False,
            initializer_range=0.02,
            gradient_checkpointing=False,
        )
        modules = [perceiver]

        # net = CAbstractor(1024, 3, 144)
        # modules = [net]

        mlp_depth = int(mlp_gelu_match.group(1))
        modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    

    raise ValueError(f'Unknown projector type: {projector_type}')
