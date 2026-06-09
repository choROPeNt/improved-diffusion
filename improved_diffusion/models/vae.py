
import torch
import torch.nn as nn
import torch.nn.functional as F



from .nn import conv_nd, conv_trans_nd, normalization, zero_module, checkpoint



class AttentionBlockSDPA(nn.Module):
    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, (x,), self.parameters(), True)
        return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        t = 1
        for s in spatial:
            t *= s

        x_flat = x.reshape(b, c, t)
        x_norm = self.norm(x_flat)  # rely on GroupNorm32 + autocast

        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, t).transpose(2, 3)  # [b,h,T,d]
        k = k.reshape(b, self.num_heads, self.head_dim, t).transpose(2, 3)
        v = v.reshape(b, self.num_heads, self.head_dim, t).transpose(2, 3)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).reshape(b, c, t)
        out = self.proj_out(out)
        return (x_flat + out).reshape(b, c, *spatial)
    

class ResBlock(nn.Module):
    def __init__(self, dims, ch):
        super().__init__()
        self.net = nn.Sequential(
            normalization( ch), nn.SiLU(),
            conv_nd(dims, ch, ch, 3, 1, 1),
            normalization(ch), nn.SiLU(),
            conv_nd(dims, ch, ch, 3, 1, 1),
        )
    def forward(self, x):
        return x + self.net(x)

class Downsample(nn.Module):
    """
    Downsample by factor 2.
    - mode="conv": strided conv (learned, standard for VAEs)
    - mode="avgpool": avg pool + 1x1 conv (optionally)
    """
    def __init__(self, dims: int, in_ch: int, out_ch: int | None = None, mode: str = "conv"):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        self.dims = dims
        self.mode = mode

        if mode == "conv":
            # k=4,s=2,p=1 gives exact /2 for even sizes (common in UNets/VAEs)
            self.op = conv_nd(
                        dims,
                        in_ch,
                        out_ch,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        )
        elif mode == "avgpool":
            if dims == 2:
                self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            elif dims == 3:
                self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
            self.proj = nn.Identity() if out_ch == in_ch else conv_nd(dims, in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError(f"mode must be 'conv' or 'avgpool', got {mode}")

    def forward(self, x):
        if self.mode == "conv":
            return self.op(x)
        x = self.pool(x)
        return self.proj(x)

class Upsample(nn.Module):
    """
    Upsample by factor 2.
    - mode="convtranspose": transposed conv (learned, standard for VAEs)
    - mode="interp": interpolate + 3x3 conv (often fewer artifacts than convtranspose)
    """
    def __init__(
        self,
        dims: int,
        in_ch: int,
        out_ch: int | None = None,
        mode: str = "interp",
        align_corners: bool = False,
    ):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch
        self.dims = dims
        self.mode = mode
        self.align_corners = align_corners

        if mode == "convtranspose":
            self.op = conv_trans_nd(dims, in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        elif mode == "interp":
            # upsample + conv is a strong default (less checkerboard)
            self.proj = conv_nd(dims, in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"mode must be 'convtranspose' or 'interp', got {mode}")

    def forward(self, x):
        if self.mode == "convtranspose":
            return self.op(x)

        # interpolate + conv
        if self.dims == 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.proj(x)



class AbstractVAE(nn.Module):
    """
    Convolutional VAE for 2D/3D data.

    Two latent modes (controlled by spatial_latent):

    spatial_latent=True  (recommended for reconstruction)
        The bottleneck feature map is projected to latent_channels via 1×1
        conv, preserving full spatial structure at the downsampled resolution.
        Latent shape: [B, latent_channels, H/ds, W/ds].  Per-pixel KL.

    spatial_latent=False  (global / legacy)
        Global average pool → linear → latent vector of size latent_dim.
        The decoder broadcasts the vector back to the spatial feature map.
        Only encodes global statistics; cannot reconstruct fine spatial detail.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 128,          # used when spatial_latent=False
        latent_channels: int = 4,       # used when spatial_latent=True
        base_channels: int = 32,
        channel_mult=(1, 2, 4),
        dims: int = 2,
        out_channels: int | None = None,
        attn_ds=(8, 16, 32),
        attn_heads=1,
        use_checkpoint=False,
        spatial_latent: bool = False,
    ):
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.spatial_latent = spatial_latent
        self.attn_ds = set(attn_ds)

        # --- Encoder ---
        chs = [base_channels * m for m in channel_mult]
        enc = [conv_nd(dims, in_channels, chs[0], kernel_size=3, stride=1, padding=1)]
        ds = 1
        for i in range(len(chs)):
            enc += [ResBlock(dims, chs[i])]
            if ds in self.attn_ds:
                enc += [AttentionBlockSDPA(chs[i], num_heads=attn_heads, use_checkpoint=use_checkpoint)]
            if i != len(chs) - 1:
                enc += [Downsample(dims, chs[i], chs[i+1], mode="conv")]
                ds *= 2
        # mid-block at the bottleneck resolution: ResBlock -> Attention -> ResBlock
        enc += [
            ResBlock(dims, chs[-1]),
            AttentionBlockSDPA(chs[-1], num_heads=attn_heads, use_checkpoint=use_checkpoint),
            ResBlock(dims, chs[-1]),
        ]
        self.encoder = nn.Sequential(*enc)

        # --- Latent projections ---
        if spatial_latent:
            self.to_mu     = conv_nd(dims, chs[-1], latent_channels, 1)
            self.to_logvar = conv_nd(dims, chs[-1], latent_channels, 1)
            self.from_z    = conv_nd(dims, latent_channels, chs[-1], 1)
        else:
            self.to_mu     = nn.Linear(chs[-1], latent_dim)
            self.to_logvar = nn.Linear(chs[-1], latent_dim)
            self.from_z    = nn.Linear(latent_dim, chs[-1])

        # --- Decoder ---
        # mirrored mid-block at the bottleneck resolution: ResBlock -> Attention -> ResBlock
        dec = [
            ResBlock(dims, chs[-1]),
            AttentionBlockSDPA(chs[-1], num_heads=attn_heads, use_checkpoint=use_checkpoint),
            ResBlock(dims, chs[-1]),
        ]
        ds_dec = ds
        for i in reversed(range(len(chs))):
            dec += [ResBlock(dims, chs[i])]
            if ds_dec in self.attn_ds:
                dec += [AttentionBlockSDPA(chs[i], num_heads=attn_heads, use_checkpoint=use_checkpoint)]
            if i != 0:
                dec += [Upsample(dims, chs[i], chs[i-1], mode="convtranspose")]
                ds_dec //= 2
        dec += [normalization(chs[0]), nn.SiLU(), conv_nd(dims, chs[0], self.out_channels, 3, 1, 1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x)
        if self.spatial_latent:
            mu     = self.to_mu(h)       # [B, latent_channels, *spatial']
            logvar = self.to_logvar(h)
            return mu, logvar
        else:
            h_vec = h.mean(dim=tuple(range(2, 2 + self.dims)))  # global avg pool
            mu     = self.to_mu(h_vec)
            logvar = self.to_logvar(h_vec)
            return mu, logvar, h.shape

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, enc_shape=None):
        if self.spatial_latent:
            h = self.from_z(z)
        else:
            b, c, *spatial = enc_shape
            h = self.from_z(z).view(z.shape[0], c, *([1] * len(spatial)))
            h = h.expand(z.shape[0], c, *spatial)
        return self.decoder(h)

    def forward(self, x):
        if self.spatial_latent:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_hat = self.decode(z)
        else:
            mu, logvar, enc_shape = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_hat = self.decode(z, enc_shape)
        return x_hat, mu, logvar, z