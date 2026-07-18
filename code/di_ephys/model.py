"""1D DeepInterpolation U-Net for extracellular ephys.

DeepInterpolation objective (Lecoq et al. 2021, Nat. Methods): predict a held-out
center time sample from a symmetric window of neighboring samples (N before +
N after), *excluding* the center sample (and an omission gap around it). Because
the target sample is never seen by the network, its noise is independent of the
model input, so the expected-loss minimizer is the noise-free signal and the
network denoises without clean targets.

For a Neuropixels probe each "frame" is the length-C vector of channel voltages
at one 30 kHz sample. Context frames are stacked along the input-channel axis of
a 1D U-Net that convolves along the probe (channel) axis and outputs a
single-frame (length-C) prediction of the center sample.

    input : (B, in_frames, C)   in_frames = number of context samples
    output: (B, 1,         C)   prediction of the center frame's C channels
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_groups(requested: int, channels: int) -> int:
    """Largest divisor of `channels` that is <= requested (GroupNorm needs it)."""
    g = min(requested, channels)
    while channels % g != 0:
        g -= 1
    return max(g, 1)


class DoubleConv1d(nn.Module):
    """(Conv1d -> Norm -> GELU) x2, with an optional residual skip.
    norm: "group" = GroupNorm (default); "none" = no normalization (SUPPORT-style)."""

    def __init__(self, in_ch: int, out_ch: int, residual: bool = True,
                 groups: int = 8, norm: str = "group"):
        super().__init__()
        self.residual = residual
        g1 = _pick_groups(groups, out_ch)
        def _norm():
            return nn.GroupNorm(g1, out_ch) if str(norm) == "group" else nn.Identity()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = _norm()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = _norm()
        self.act = nn.GELU()
        self.proj = (nn.Conv1d(in_ch, out_ch, 1, bias=False)
                     if (residual and in_ch != out_ch) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if self.residual:
            if self.proj is not None:
                identity = self.proj(identity)
            h = h + identity
        return self.act(h)


class LayerNorm1d(nn.Module):
    """LayerNorm over features at each probe position."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class SimpleGate1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock1d(nn.Module):
    """One-dimensional NAFNet-style restoration block."""

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        expanded = channels * expansion
        self.norm1 = LayerNorm1d(channels)
        self.expand = nn.Conv1d(channels, expanded, 1)
        self.depthwise = nn.Conv1d(
            expanded, expanded, 3, padding=1, groups=expanded
        )
        self.gate1 = SimpleGate1d()
        gated = expanded // 2
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(gated, gated, 1),
        )
        self.project = nn.Conv1d(gated, channels, 1)

        self.norm2 = LayerNorm1d(channels)
        self.ffn_expand = nn.Conv1d(channels, expanded, 1)
        self.gate2 = SimpleGate1d()
        self.ffn_project = nn.Conv1d(gated, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.depthwise(self.expand(self.norm1(x)))
        h = self.gate1(h)
        h = self.project(h * self.channel_attention(h))
        y = x + self.beta * h
        h = self.ffn_project(self.gate2(self.ffn_expand(self.norm2(y))))
        return y + self.gamma * h


class NAFStage1d(nn.Module):
    """Project to the stage width, then apply one NAF block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.project = (
            nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        )
        self.block = NAFBlock1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.project(x))


def temporal_stage(in_ch: int, out_ch: int, residual: bool, norm: str,
                   block_type: str) -> nn.Module:
    name = str(block_type).lower()
    if name == "doubleconv":
        return DoubleConv1d(in_ch, out_ch, residual, norm=norm)
    if name == "naf":
        return NAFStage1d(in_ch, out_ch)
    raise ValueError(f"unsupported temporal block: {block_type}")


class Down1d(nn.Module):
    def __init__(self, in_ch, out_ch, residual=True, norm="group",
                 block_type="doubleconv"):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = temporal_stage(in_ch, out_ch, residual, norm, block_type)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up1d(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, residual=True, norm="group",
                 block_type="doubleconv"):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = temporal_stage(
            in_ch // 2 + skip_ch, out_ch, residual, norm, block_type
        )

    def forward(self, x, skip):
        x = self.up(x)
        dl = skip.shape[-1] - x.shape[-1]          # pad if odd length mismatch
        if dl:
            x = F.pad(x, [dl // 2, dl - dl // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionBlock1d(nn.Module):
    """Multi-head self-attention over probe positions (bottleneck only)."""

    def __init__(self, channels: int, num_heads: int = 4, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(_pick_groups(groups, channels), channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        y = self.norm(x).transpose(1, 2)           # (B, L, C)
        y, _ = self.mha(y, y, y, need_weights=False)
        y = y.transpose(1, 2)                       # (B, C, L)
        return x + y


class ConvHole1D(nn.Conv1d):
    """Conv1d with the CENTER kernel tap forced to zero (a 1-D architectural
    blind spot along the probe axis; SUPPORT/Laine-style). The output at channel
    c never depends on the input at channel c. Stacked with a geometric dilation
    schedule (dilation = 2**d) the compounded receptive field reaches every
    channel offset EXCEPT 0, so the prediction of a channel is provably
    independent of that channel's own (noisy) value.

    NO norm that mixes across the probe axis may follow these convs (it would leak
    the center channel back in via the statistics); only pointwise ops are safe.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, bias=True):
        pad = dilation * (kernel_size // 2)
        super().__init__(in_ch, out_ch, kernel_size, stride=1,
                         padding=pad, dilation=dilation, bias=bias)
        hole = torch.ones(1, 1, kernel_size)
        hole[0, 0, kernel_size // 2] = 0.0
        self.register_buffer("hole", hole)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(x, self.weight * self.hole, self.bias)


class BlindSpotBranch1d(nn.Module):
    """Probe-axis blind-spot subnet over the CENTER frame.

    A stack of dilated hole-convolutions (dilation = 2**d) with pointwise GELU
    only -- no norm -- so each channel's feature excludes that channel's own
    center-frame value. Emits (B, out_ch, C) features fused per-channel with the
    temporal (center-excluded) U-Net path.
    """

    def __init__(self, out_ch: int = 64, depth: int = 5, in_ch: int = 1):
        super().__init__()
        layers = []
        c_in = in_ch
        for d in range(depth):
            layers.append(ConvHole1D(c_in, out_ch, kernel_size=3, dilation=2 ** d))
            layers.append(nn.GELU())
            c_in = out_ch
        self.net = nn.Sequential(*layers)
        self.out_ch = out_ch

    def forward(self, center: torch.Tensor) -> torch.Tensor:   # (B, 1, C)
        return self.net(center)


class SupportBlindSpotBranch1d(nn.Module):
    """SUPPORT-style probe-axis blind-spot subnet. On top of the plain dilated
    hole-conv stack it optionally: (a) re-injects the centre input at every layer
    via a per-layer scalar (DENSE skips), (b) injects the temporal U-Net feature
    at the first layer (STAGING), and (c) runs a parallel 5-kernel stack for
    MULTI-SCALE coverage.

    Blind spot is preserved: every dense skip is added BEFORE a hole-conv (whose
    centre tap is zero) and the geometric dilations (2**d for the 3-kernel stack,
    3**d for the 5-kernel stack) forbid any path of non-zero taps from summing
    back to offset 0; the injected U-Net feature is derived from the
    centre-EXCLUDED neighbour frames, so it carries no info about the target.
    """

    def __init__(self, in_ch: int, out_ch: int = 64, depth: int = 5,
                 dense: bool = True, stage_ch: int = 0, multiscale: bool = False):
        super().__init__()
        self.dense = bool(dense)
        self.multiscale = bool(multiscale)
        self.act = nn.GELU()
        self.convs3 = nn.ModuleList(
            [ConvHole1D(in_ch if d == 0 else out_ch, out_ch, kernel_size=3,
                        dilation=2 ** d) for d in range(depth)])
        depth5 = max(1, depth - 2) if multiscale else 0
        self.convs5 = (nn.ModuleList(
            [ConvHole1D(in_ch if d == 0 else out_ch, out_ch, kernel_size=5,
                        dilation=3 ** d) for d in range(depth5)])
            if multiscale else None)
        if self.dense:
            self.dense_proj = nn.Conv1d(in_ch, out_ch, 1)
            self.dense_scale3 = nn.Parameter(torch.ones(max(1, depth - 1)))
            if multiscale:
                self.dense_scale5 = nn.Parameter(torch.ones(max(1, depth5 - 1)))
        self.stage = nn.Conv1d(stage_ch, out_ch, 1) if stage_ch > 0 else None
        self.out_total = out_ch * (2 if multiscale else 1)

    def _run(self, convs, center, cproj, scales, u_feat):
        x = center
        for c, conv in enumerate(convs):
            if c > 0 and self.dense:
                x = x + scales[c - 1] * cproj             # dense skip BEFORE hole-conv
            x = self.act(conv(x))                         # hole-conv (centre tap = 0)
            if c == 0 and u_feat is not None:
                x = x + u_feat                            # stage U-Net feature (blind-safe)
        return x

    def forward(self, center, u=None):                    # (B, in_ch, H)
        cproj = self.dense_proj(center) if self.dense else None
        u_feat = self.stage(u) if (self.stage is not None and u is not None) else None
        out = self._run(self.convs3, center, cproj,
                        getattr(self, "dense_scale3", None), u_feat)
        if self.multiscale:
            out5 = self._run(self.convs5, center, cproj,
                             getattr(self, "dense_scale5", None), u_feat)
            out = torch.cat([out, out5], dim=1)
        return out


class DeepInterpUNet1D(nn.Module):
    """Configurable 1D U-Net mapping (B, in_frames, C) -> (B, 1, C)."""

    def __init__(self, in_frames: int, base_channels: int = 32, depth: int = 4,
                 residual: bool = True, attention_bottleneck: bool = False,
                 out_activation: str = "none", out_channels: int = 1,
                 norm: str = "group", block_type: str = "doubleconv"):
        super().__init__()
        assert in_frames >= 2, "need at least 2 context frames"
        assert depth >= 1
        self.in_frames = in_frames
        self.depth = depth

        chans = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.stem = temporal_stage(
            in_frames, chans[0], residual, norm, block_type
        )
        self.downs = nn.ModuleList(
            [Down1d(chans[i], chans[i + 1], residual, norm=norm,
                    block_type=block_type) for i in range(depth)])
        self.attn = (AttentionBlock1d(chans[-1]) if attention_bottleneck
                     else nn.Identity())
        self.ups = nn.ModuleList(
            [Up1d(chans[i + 1], chans[i], chans[i], residual, norm=norm,
                  block_type=block_type)
             for i in reversed(range(depth))])
        self.head = nn.Conv1d(chans[0], out_channels, 1)
        self._act = {"none": nn.Identity(), "tanh": nn.Tanh(),
                     "relu": nn.ReLU()}[out_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"expected (B,C,L), got {tuple(x.shape)}"
        assert x.shape[1] == self.in_frames, (
            f"expected {self.in_frames} context frames, got {x.shape[1]}")
        skips = []
        h = self.stem(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)
        h = self.attn(h)
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            h = up(h, skip)
        return self._act(self.head(h))


class BlindSpotDeepInterp1D(nn.Module):
    """SUPPORT-style ephys denoiser: a temporal U-Net over the center-EXCLUDED
    neighbor frames, fused per-channel with a probe-axis blind-spot subnet over
    the center frame.

    Input layout (B, in_frames, C): the LAST frame-channel is the (real) center
    frame; the first in_frames-1 are the temporal neighbors. The U-Net sees only
    the neighbors (so it can never leak the center's own noise); the blind-spot
    branch sees the center frame but, by construction, its output at channel c is
    independent of the center's value at c. Fusion is pointwise (1x1 over the
    probe axis), which cannot reintroduce a channel's own value -- so every output
    channel is a statistically unbiased prediction of the center frame.
    """

    def __init__(self, in_frames: int, base_channels: int = 32, depth: int = 4,
                 residual: bool = True, attention_bottleneck: bool = False,
                 out_activation: str = "none", bs_channels: int = 64,
                 bs_depth: int = 5, fuse_channels: int = 64,
                 temporal_block: str = "doubleconv"):
        super().__init__()
        assert in_frames >= 3, "need >= 2 neighbors + 1 center frame"
        self.in_frames = in_frames
        self.unet = DeepInterpUNet1D(
            in_frames=in_frames - 1, base_channels=base_channels, depth=depth,
            residual=residual, attention_bottleneck=attention_bottleneck,
            out_activation="none", block_type=temporal_block)
        self.bsnet = BlindSpotBranch1d(out_ch=bs_channels, depth=bs_depth)
        # pointwise fusion only (preserves the blind spot)
        self.fuse = nn.Sequential(
            nn.Conv1d(1 + bs_channels, fuse_channels, 1), nn.GELU(),
            nn.Conv1d(fuse_channels, fuse_channels, 1), nn.GELU(),
            nn.Conv1d(fuse_channels, 1, 1),
        )
        self._act = {"none": nn.Identity(), "tanh": nn.Tanh(),
                     "relu": nn.ReLU()}[out_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[1] == self.in_frames, (
            f"expected (B,{self.in_frames},C), got {tuple(x.shape)}")
        ctx = x[:, :-1]                       # temporal neighbors (center excluded)
        center = x[:, -1:]                    # center frame
        u = self.unet(ctx)                    # (B, 1, C)   temporal prediction
        b = self.bsnet(center)                # (B, bs_ch, C)  blind-spot features
        return self._act(self.fuse(torch.cat([u, b], dim=1)))


# ----------------------------------------------------------------------------
# Geometry-aware 2D variant. Neuropixels contacts sit on a narrow 2-D lattice
# (depth x width), not a 1-D line, so a channel's true neighbours live in 2-D.
# These layers convolve over the real probe grid (built from the channels'
# physical x/y locations) instead of the channel-index axis. The probe is long
# and thin, so pooling is along DEPTH only; width is kept full through the net.
# ----------------------------------------------------------------------------


class DoubleConv2d(nn.Module):
    """(Conv2d -> GroupNorm -> GELU) x2 over the probe grid, optional residual."""

    def __init__(self, in_ch, out_ch, residual=True, groups=8):
        super().__init__()
        self.residual = residual
        g1 = _pick_groups(groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g1, out_ch)
        self.act = nn.GELU()
        self.proj = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if (residual and in_ch != out_ch) else None)

    def forward(self, x):
        identity = x
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if self.residual:
            if self.proj is not None:
                identity = self.proj(identity)
            h = h + identity
        return self.act(h)


class Down2d(nn.Module):
    def __init__(self, in_ch, out_ch, residual=True):
        super().__init__()
        self.pool = nn.MaxPool2d((2, 1))                 # pool DEPTH only
        self.conv = DoubleConv2d(in_ch, out_ch, residual)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up2d(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, residual=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, (2, 1), stride=(2, 1))
        self.conv = DoubleConv2d(in_ch // 2 + skip_ch, out_ch, residual)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]                # pad depth if odd
        if dh:
            x = F.pad(x, [0, 0, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionBlock2d(nn.Module):
    """Multi-head self-attention over all grid positions (bottleneck only)."""

    def __init__(self, channels, num_heads=4, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(_pick_groups(groups, channels), channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)      # (B, H*W, C)
        y, _ = self.mha(y, y, y, need_weights=False)
        y = y.transpose(1, 2).view(b, c, h, w)
        return x + y


class ConvHole2D(nn.Conv2d):
    """Conv2d with the CENTER kernel tap forced to zero: the output at grid cell
    (r, c) never depends on the input at (r, c). Stacked with per-axis dilation
    = 2**d on BOTH axes, any tap path's signed offset is a sum of distinct
    powers of two on each axis -- zero only for the empty path -- so a cell's own
    (noisy) value can never be reintroduced (a 2-D blind spot on the probe grid).
    Only pointwise ops may follow (norm across the grid would leak it back in).
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, bias=True):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        pad = (dilation[0] * (kernel_size[0] // 2), dilation[1] * (kernel_size[1] // 2))
        super().__init__(in_ch, out_ch, kernel_size, stride=1,
                         padding=pad, dilation=dilation, bias=bias)
        hole = torch.ones(1, 1, kernel_size[0], kernel_size[1])
        hole[0, 0, kernel_size[0] // 2, kernel_size[1] // 2] = 0.0
        self.register_buffer("hole", hole)

    def forward(self, x):
        return self._conv_forward(x, self.weight * self.hole, self.bias)


class BlindSpotBranch2d(nn.Module):
    """Probe-grid blind-spot subnet over the CENTER frame: stacked dilated
    hole-convs with pointwise GELU only. `depth_only` uses cheap (3x1) hole convs
    dilated along depth only (as cheap as the 1-D branch; the narrow width axis
    is then mixed by the center-excluded U-Net) instead of (3x3) hole convs."""

    def __init__(self, out_ch: int = 64, depth: int = 5, depth_only: bool = False):
        super().__init__()
        layers = []
        c_in = 1
        for d in range(depth):
            k = (3, 1) if depth_only else 3
            dil = (2 ** d, 1) if depth_only else 2 ** d
            layers.append(ConvHole2D(c_in, out_ch, kernel_size=k, dilation=dil))
            layers.append(nn.GELU())
            c_in = out_ch
        self.net = nn.Sequential(*layers)
        self.out_ch = out_ch

    def forward(self, center):                           # (B, 1, H, W)
        return self.net(center)


class _Grid(nn.Module):
    """Scatter (B, F, C) channels onto a (H, W) probe grid and gather back.
    Cells with no contact (e.g. the empty half of an NP1 checkerboard) stay
    zero and are never gathered, so they cannot affect a real channel's loss."""

    def __init__(self, flat_pos, H, W):
        super().__init__()
        self.H, self.W = int(H), int(W)
        self.register_buffer("flat_pos", torch.as_tensor(flat_pos, dtype=torch.long))

    def scatter(self, x):                                # (B, F, C) -> (B, F, H, W)
        b, f, _ = x.shape
        g = x.new_zeros(b, f, self.H * self.W)
        g[:, :, self.flat_pos] = x
        return g.view(b, f, self.H, self.W)

    def gather(self, g):                                 # (B, F, H, W) -> (B, F, C)
        b, f = g.shape[:2]
        return g.reshape(b, f, self.H * self.W)[:, :, self.flat_pos]


class DeepInterpUNet2D(nn.Module):
    """U-Net over the probe grid (B, in_frames, H, W) -> (B, 1, H, W); pools
    along depth only so the narrow width axis is preserved."""

    def __init__(self, in_frames, H, W, base_channels=32, depth=4, residual=True,
                 attention_bottleneck=False):
        super().__init__()
        assert in_frames >= 1 and depth >= 1
        self.in_frames = in_frames
        chans = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.stem = DoubleConv2d(in_frames, chans[0], residual)
        self.downs = nn.ModuleList(
            [Down2d(chans[i], chans[i + 1], residual) for i in range(depth)])
        self.attn = (AttentionBlock2d(chans[-1]) if attention_bottleneck
                     else nn.Identity())
        self.ups = nn.ModuleList(
            [Up2d(chans[i + 1], chans[i], chans[i], residual)
             for i in reversed(range(depth))])
        self.head = nn.Conv2d(chans[0], 1, 1)

    def forward(self, x):
        skips = []
        h = self.stem(x)
        skips.append(h)
        for down in self.downs:
            h = down(h)
            skips.append(h)
        h = self.attn(h)
        for i, up in enumerate(self.ups):
            h = up(h, skips[-(i + 2)])
        return self.head(h)


class DeepInterpGrid2D(nn.Module):
    """Non-blind geometry-aware model: scatter -> 2-D U-Net -> gather."""

    def __init__(self, in_frames, flat_pos, H, W, base_channels=32, depth=4,
                 residual=True, attention_bottleneck=False, out_activation="none"):
        super().__init__()
        self.in_frames = in_frames
        self.grid = _Grid(flat_pos, H, W)
        self.unet = DeepInterpUNet2D(in_frames, H, W, base_channels, depth,
                                     residual, attention_bottleneck)
        self._act = {"none": nn.Identity(), "tanh": nn.Tanh(),
                     "relu": nn.ReLU()}[out_activation]

    def forward(self, x):                                # (B, in_frames, C)->(B,1,C)
        y = self._act(self.unet(self.grid.scatter(x)))
        return self.grid.gather(y)


class BlindSpotDeepInterp2D(nn.Module):
    """Geometry-aware SUPPORT-style denoiser: a 2-D temporal U-Net over the
    center-EXCLUDED neighbour frames fused per-cell with a probe-grid blind-spot
    subnet over the center frame. Output(r,c) is independent of center(r,c)."""

    def __init__(self, in_frames, flat_pos, H, W, base_channels=32, depth=4,
                 residual=True, attention_bottleneck=False, out_activation="none",
                 bs_channels=64, bs_depth=5, fuse_channels=64, bs_depth_only=False):
        super().__init__()
        assert in_frames >= 3, "need >= 2 neighbors + 1 center frame"
        self.in_frames = in_frames
        self.grid = _Grid(flat_pos, H, W)
        self.unet = DeepInterpUNet2D(in_frames - 1, H, W, base_channels, depth,
                                     residual, attention_bottleneck)
        self.bsnet = BlindSpotBranch2d(out_ch=bs_channels, depth=bs_depth,
                                       depth_only=bs_depth_only)
        self.fuse = nn.Sequential(                       # pointwise only (preserves blind spot)
            nn.Conv2d(1 + bs_channels, fuse_channels, 1), nn.GELU(),
            nn.Conv2d(fuse_channels, fuse_channels, 1), nn.GELU(),
            nn.Conv2d(fuse_channels, 1, 1),
        )
        self._act = {"none": nn.Identity(), "tanh": nn.Tanh(),
                     "relu": nn.ReLU()}[out_activation]

    def forward(self, x):                                # (B, in_frames, C)->(B,1,C)
        g = self.grid.scatter(x)
        u = self.unet(g[:, :-1])                         # neighbours (center excluded)
        b = self.bsnet(g[:, -1:])                        # center frame, blind spot
        y = self._act(self.fuse(torch.cat([u, b], dim=1)))
        return self.grid.gather(y)


class FoldDeepInterp1D(nn.Module):
    """Geometry-aware but 1-D-fast denoiser: scatter channels onto the (H, W)
    probe grid, FOLD the W columns into the feature axis, and run a 1-D U-Net
    along DEPTH (length H). Compute is close to the 1-D model (W folds into the
    channel dim while the conv length is only H), but neighbourhoods follow the
    real probe depth and the columns mix through the channel dimension. The
    blind spot is along depth: the center branch excludes the whole row (all
    columns), hence the target cell."""

    def __init__(self, in_frames, flat_pos, H, W, base_channels=32, depth=4,
                 residual=True, attention_bottleneck=False, out_activation="none",
                 bs_channels=64, bs_depth=5, fuse_channels=64, blind_spot=True,
                 bs_frames=1, temporal_mult=1,
                 bs_stage=False, bs_dense=False, bs_multiscale=False, norm="group",
                 temporal_block="doubleconv"):
        super().__init__()
        self.grid = _Grid(flat_pos, H, W)
        self.W = int(W)
        self.blind_spot = bool(blind_spot)
        self.bs_frames = int(bs_frames) if self.blind_spot else 0
        # temporal hand-off width: the U-Net emits W*temporal_mult feature maps
        # into the pointwise fuse head (temporal_mult=1 = the champion's committed
        # W-column prediction; >1 gives the head richer temporal features to decode
        # diverse spike shapes). Only meaningful with the blind-spot fuse; without
        # it the U-Net output must stay W (the prediction itself).
        self.temporal_mult = int(temporal_mult) if self.blind_spot else 1
        u_out = self.W * self.temporal_mult
        nf = in_frames - self.bs_frames
        self.unet = DeepInterpUNet1D(
            in_frames=nf * self.W, base_channels=base_channels, depth=depth,
            residual=residual, attention_bottleneck=attention_bottleneck,
            out_activation="none", out_channels=u_out, norm=norm,
            block_type=temporal_block)
        self.bs_stage = bool(bs_stage) and self.blind_spot
        self.bs_dense = bool(bs_dense) and self.blind_spot
        self.bs_multiscale = bool(bs_multiscale) and self.blind_spot
        self.support = self.bs_stage or self.bs_dense or self.bs_multiscale
        if self.blind_spot:
            bs_in = self.W * self.bs_frames
            if self.support:
                self.bsnet = SupportBlindSpotBranch1d(
                    in_ch=bs_in, out_ch=bs_channels, depth=bs_depth,
                    dense=self.bs_dense, stage_ch=(u_out if self.bs_stage else 0),
                    multiscale=self.bs_multiscale)
                fuse_bs = self.bsnet.out_total
            else:
                self.bsnet = BlindSpotBranch1d(out_ch=bs_channels, depth=bs_depth,
                                               in_ch=bs_in)
                fuse_bs = bs_channels
            self.fuse = nn.Sequential(
                nn.Conv1d(u_out + fuse_bs, fuse_channels, 1), nn.GELU(),
                nn.Conv1d(fuse_channels, fuse_channels, 1), nn.GELU(),
                nn.Conv1d(fuse_channels, self.W, 1))
        self._act = {"none": nn.Identity(), "tanh": nn.Tanh(),
                     "relu": nn.ReLU()}[out_activation]

    def _fold(self, g):                        # (B, F, H, W) -> (B, F*W, H)
        b, f, h, w = g.shape
        return g.permute(0, 1, 3, 2).reshape(b, f * w, h)

    def _unfold(self, y):                      # (B, W, H) -> (B, 1, H, W)
        return y.permute(0, 2, 1).unsqueeze(1)

    def forward(self, x):                      # (B, in_frames, C) -> (B, 1, C)
        g = self.grid.scatter(x)
        if self.blind_spot:
            k = self.bs_frames
            u = self.unet(self._fold(g[:, :-k]))           # neighbours -> (B, u_out, H)
            if self.support:
                b = self.bsnet(self._fold(g[:, -k:]), u if self.bs_stage else None)
            else:
                b = self.bsnet(self._fold(g[:, -k:]))      # k centre frames -> (B, bs_ch, H)
            y = self.fuse(torch.cat([u, b], dim=1))        # (B, W, H)
        else:
            y = self.unet(self._fold(g))                   # (B, W, H)
        return self.grid.gather(self._act(self._unfold(y)))


class OrigEphysUNet2D(nn.Module):
    """Faithful re-implementation of DeepInterpolation's ORIGINAL ephys U-Net
    (`unet_single_ephys_1024`, Lecoq et al. 2021, network_collection.py): a
    5-level 2-D U-Net over the probe grid with a SINGLE Conv2d + ReLU per level
    (no normalization, no residual), concatenative skips, widths
    64->128->256->512->1024. Kernels are (2x2) at the top level and (3x1) in the
    interior; pooling is (2x2) at the top and (2x1) (depth-only) below. Linear
    output. Maps (B, in_frames, H, W) -> (B, 1, H, W). Provided for a controlled
    architecture comparison against the `fold` champion (same data/frames/loss).
    """

    def __init__(self, in_frames: int, base_channels: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth
        c = [base_channels * (2 ** i) for i in range(depth + 1)]      # [64,128,256,512,1024]
        pk = [(2, 2) if i == 0 else (2, 1) for i in range(depth)]     # pool per level
        # encoder: conv1 (2x2) in_frames->c0; conv2..conv_{depth+1} (3x1) c[i-1]->c[i]
        self.enc = nn.ModuleList([nn.Conv2d(in_frames, c[0], (2, 2), padding="same")])
        for i in range(1, depth + 1):
            self.enc.append(nn.Conv2d(c[i - 1], c[i], (3, 1), padding="same"))
        self.pools = nn.ModuleList([nn.MaxPool2d(pk[i]) for i in range(depth)])
        # decoder: mirror; upsample (nearest) then single conv on concat(skip, up)
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(depth)):                             # i = depth-1 .. 0
            self.ups.append(nn.Upsample(scale_factor=pk[i], mode="nearest"))
            dk = (2, 2) if i == 0 else (3, 1)
            self.dec.append(nn.Conv2d(c[i + 1] + c[i], c[i], dk, padding="same"))
        self.head = nn.Conv2d(c[0], 1, (1, 1))

    def forward(self, x):
        skips = []
        h = F.relu(self.enc[0](x))                                   # conv1
        skips.append(h)
        for i in range(self.depth):
            h = F.relu(self.enc[i + 1](self.pools[i](h)))            # conv2..conv5
            if i < self.depth - 1:
                skips.append(h)                                      # conv2..conv4 (conv5 = bottleneck)
        for j, (up, dec) in enumerate(zip(self.ups, self.dec)):
            h = up(h)
            skip = skips[-(j + 1)]
            dh, dw = skip.shape[-2] - h.shape[-2], skip.shape[-1] - h.shape[-1]
            if dh or dw:
                h = F.pad(h, [0, max(dw, 0), 0, max(dh, 0)])
            h = F.relu(dec(torch.cat([skip, h], dim=1)))
        return self.head(h)


class DeepInterpGridOrig(nn.Module):
    """Original DeepInterpolation ephys geometry: scatter channels onto the probe
    grid, run the faithful `OrigEphysUNet2D`, gather. Pure temporal-hole DI (no
    channel blind spot) -- the input frames are the center-EXCLUDED neighbors."""

    def __init__(self, in_frames, flat_pos, H, W, base_channels=64, depth=4,
                 out_activation="none"):
        super().__init__()
        self.in_frames = in_frames
        self.grid = _Grid(flat_pos, H, W)
        self.unet = OrigEphysUNet2D(in_frames, base_channels, depth)
        self._act = {"none": nn.Identity(), "tanh": nn.Tanh(),
                     "relu": nn.ReLU()}[out_activation]

    def forward(self, x):                                # (B, in_frames, C) -> (B, 1, C)
        y = self._act(self.unet(self.grid.scatter(x)))
        return self.grid.gather(y)


def build_model(cfg: dict, in_frames: int, grid=None):
    geometry = str(cfg.get("geometry", "1d")).lower()
    if geometry == "orig":
        if grid is None:
            raise ValueError("geometry='orig' requires a channel grid (flat_pos, H, W)")
        flat_pos, H, W = grid["flat_pos"], grid["H"], grid["W"]
        return DeepInterpGridOrig(
            in_frames, flat_pos, H, W,
            base_channels=int(cfg.get("base_channels", 64)),
            depth=int(cfg.get("depth", 4)),
            out_activation=cfg.get("out_activation", "none"))
    if geometry in ("2d", "fold"):
        if grid is None:
            raise ValueError(f"geometry='{geometry}' requires a channel grid (flat_pos, H, W)")
        flat_pos, H, W = grid["flat_pos"], grid["H"], grid["W"]
        common = dict(
            base_channels=int(cfg.get("base_channels", 32)),
            depth=int(cfg.get("depth", 4)),
            residual=bool(cfg.get("residual", True)),
            attention_bottleneck=bool(cfg.get("attention_bottleneck", False)),
            out_activation=cfg.get("out_activation", "none"))
        if geometry == "fold":
            return FoldDeepInterp1D(
                in_frames, flat_pos, H, W,
                bs_channels=int(cfg.get("bs_channels", 64)),
                bs_depth=int(cfg.get("bs_depth", 5)),
                fuse_channels=int(cfg.get("fuse_channels", 64)),
                blind_spot=bool(cfg.get("blind_spot", False)),
                bs_frames=int(cfg.get("bs_frames", 1)),
                temporal_mult=int(cfg.get("temporal_mult", 1)),
                bs_stage=bool(cfg.get("bs_stage", False)),
                bs_dense=bool(cfg.get("bs_dense", False)),
                bs_multiscale=bool(cfg.get("bs_multiscale", False)),
                norm=str(cfg.get("norm", "group")),
                temporal_block=str(cfg.get("temporal_block", "doubleconv")),
                **common)
        if bool(cfg.get("blind_spot", False)):
            return BlindSpotDeepInterp2D(
                in_frames, flat_pos, H, W,
                bs_channels=int(cfg.get("bs_channels", 64)),
                bs_depth=int(cfg.get("bs_depth", 5)),
                fuse_channels=int(cfg.get("fuse_channels", 64)),
                bs_depth_only=bool(cfg.get("bs2d_depth_only", False)),
                **common)
        return DeepInterpGrid2D(in_frames, flat_pos, H, W, **common)

    if bool(cfg.get("blind_spot", False)):
        return BlindSpotDeepInterp1D(
            in_frames=in_frames,
            base_channels=int(cfg.get("base_channels", 32)),
            depth=int(cfg.get("depth", 4)),
            residual=bool(cfg.get("residual", True)),
            attention_bottleneck=bool(cfg.get("attention_bottleneck", False)),
            out_activation=cfg.get("out_activation", "none"),
            bs_channels=int(cfg.get("bs_channels", 64)),
            bs_depth=int(cfg.get("bs_depth", 5)),
            fuse_channels=int(cfg.get("fuse_channels", 64)),
            temporal_block=str(cfg.get("temporal_block", "doubleconv")),
        )
    return DeepInterpUNet1D(
        in_frames=in_frames,
        base_channels=int(cfg.get("base_channels", 32)),
        depth=int(cfg.get("depth", 4)),
        residual=bool(cfg.get("residual", True)),
        attention_bottleneck=bool(cfg.get("attention_bottleneck", False)),
        out_activation=cfg.get("out_activation", "none"),
        block_type=str(cfg.get("temporal_block", "doubleconv")),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
