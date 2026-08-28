"""
dit.py — DiT backbone with 1D, 2D, and video support.

Changes vs. original
─────────────────────
New classes
  PatchEmbed1DVideo   (B, F, C, N)     → (B, F·S, D)
  PatchEmbed2DVideo   (B, F, C, H, W)  → (B, F·S, D)
  SpatialDiTBlock     full DiT block; folds frames into batch → (B·F, S, D)
  TemporalDiTBlock    DiT block that attends across frames    → (B·S, F, D)

DiT gains two new constructor args
  num_frames : int  = 1      activates video mode when > 1
  factorize  : bool = False  alternate spatial/temporal blocks instead of
                             full space-time attention

All image behaviour (num_frames=1) is byte-for-byte identical to the original.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import repeat, rearrange, pack, unpack
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Mlp

from .attend import Attention, VisionRotaryEmbeddingFast, LinearAttention, WindowAttention, PhysicsAttention
from .basic_modules import SwiGLUFFN

import math
from functools import partial


# ─── helpers ────────────────────────────────────────────────────────────────────

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern=None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')
    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim=-1)
    parallel = (x * unit).sum(dim=-1, keepdim=True) * unit
    orthogonal = x - parallel
    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)


# ─── timestep / condition embedders (unchanged) ─────────────────────────────────

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256, bias=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ConditionEmbedder(nn.Module):
    """Embeds conditions with optional CFG dropout."""
    def __init__(self, cond_dim, hidden_size, dropout_prob):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.null_classes_emb = nn.Parameter(torch.randn(cond_dim))
        self.dropout_prob = dropout_prob

    def token_drop(self, cond_variables, force_drop_ids=None):
        batch = cond_variables.shape[0]
        if force_drop_ids is None:
            drop_ids = torch.rand(cond_variables.shape[0], device=cond_variables.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)
        return torch.where(rearrange(drop_ids, "b -> b 1"), null_classes_emb, cond_variables)

    def forward(self, cond_variables, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond_variables = self.token_drop(cond_variables, force_drop_ids)
        return self.mlp(cond_variables)


# ─── transformer blocks (unchanged) ─────────────────────────────────────────────

class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, bias=True,
                 use_swiglu=False, attn_type="vanilla", qk_norm=False,
                 num_experts=None, num_experts_per_tok=None, **attn_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=bias)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=bias)

        if attn_type == "vanilla":
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=bias,
                                  proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        elif attn_type == "linear":
            self.attn = LinearAttention(hidden_size, num_heads=num_heads, qkv_bias=bias,
                                        proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        elif attn_type == "physics":
            self.attn = PhysicsAttention(hidden_size, num_heads=num_heads, qkv_bias=bias,
                                         proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        else:
            self.attn = None

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if num_experts is not None and num_experts_per_tok is not None:
            pass
            # self.mlp = SparseMoeBlock(embed_dim=hidden_size, mlp_ratio=mlp_ratio,
            #                           num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        elif use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim), bias=bias)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                           act_layer=approx_gelu, drop=0, bias=bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=bias)
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x.contiguous()
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def window_partition(x, window_size):
    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    return x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)


class WindowBlock(DiTBlock):
    def __init__(self, hidden_size, num_heads, window_size, shift_size=0, *args, **kwargs):
        super().__init__(hidden_size, num_heads, *args, **kwargs)
        self.attn = WindowAttention(hidden_size, window_size=window_size,
                                    num_heads=num_heads, qkv_bias=False)
        self.window_size = window_size

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows).view(x.shape)
        x = x + gate_msa.unsqueeze(1) * attn_windows
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=bias)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ─── 1-D patch embedder & final layer (unchanged) ────────────────────────────────

class PatchEmbed1D(nn.Module):
    """(B, C, N) → (B, num_patches, D)"""
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)


class FinalLayer1D(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=bias)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW: Video patch embedders
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEmbed1DVideo(nn.Module):
    """
    (B, F, C, N) → (B, F * num_spatial_patches, D)

    Tokens are laid out frame-first:
        [f0_s0, f0_s1, …, f0_sS,  f1_s0, …,  f(F-1)_sS]
    The same layout is assumed by _video_pos_embed() and unpatchify().
    """
    def __init__(self, seq_len, num_frames, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_frames          = num_frames
        self.num_spatial_patches = seq_len // patch_size
        self.num_patches         = num_frames * self.num_spatial_patches
        self.proj = nn.Conv1d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                          # x: (B, F, C, N)
        B, F, C, N = x.shape
        x = x.reshape(B * F, C, N)
        x = self.proj(x).transpose(1, 2)          # (B*F, S, D)
        return x.reshape(B, self.num_patches, -1)  # (B, F*S, D)


class PatchEmbed2DVideo(nn.Module):
    """
    (B, F, C, H, W) → (B, F * num_spatial_patches, D)
    Reuses timm's PatchEmbed for the per-frame spatial projection.
    """
    def __init__(self, img_size, num_frames, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_frames          = num_frames
        self._spatial_embed      = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_spatial_patches = self._spatial_embed.num_patches
        self.num_patches         = num_frames * self.num_spatial_patches
        # expose .proj so initialize_weights() can reach it via the same path
        self.proj                = self._spatial_embed.proj

    def forward(self, x):                          # x: (B, F, C, H, W)
        B, F, C, H, W = x.shape
        x = x.reshape(B * F, C, H, W)
        x = self._spatial_embed(x)                # (B*F, S, D)
        return x.reshape(B, self.num_patches, -1)  # (B, F*S, D)


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW: Factorised spatial / temporal blocks
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialDiTBlock(DiTBlock):
    """
    Standard DiT attention over the spatial dimension only.

    Accepts the full video token sequence  (B, F·S, D)  but internally folds
    frames into the batch so attention is limited to within-frame patches:
        (B, F·S, D)  →  (B·F, S, D)  →  attention  →  (B, F·S, D)

    The conditioning vector c (B, D) is tiled to (B·F, D) accordingly.
    Spatial RoPE is forwarded as usual.
    """
    def __init__(self, hidden_size, num_heads, *args, num_frames=1, **kwargs):
        super().__init__(hidden_size, num_heads, *args, **kwargs)
        self.num_frames = num_frames

    def forward(self, x, c, feat_rope=None):
        B, FS, D = x.shape
        F, S = self.num_frames, FS // self.num_frames

        # fold frames into batch dimension
        x_s = x.reshape(B * F, S, D)
        c_s = c.unsqueeze(1).expand(-1, F, -1).reshape(B * F, D)

        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_s).chunk(6, dim=1)
        x_s = x_s + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x_s), shift_msa, scale_msa), rope=feat_rope)
        x_s = x_s.contiguous()
        x_s = x_s + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x_s), shift_mlp, scale_mlp))

        return x_s.reshape(B, FS, D)


class TemporalDiTBlock(DiTBlock):
    """
    DiT attention across the temporal (frame) dimension.

    Accepts (B, F·S, D) and transposes to make frames the sequence axis:
        (B, F·S, D)  →  (B·S, F, D)  →  attention  →  (B, F·S, D)

    Temporal positional ordering comes from the additive temporal pos-embed
    added in forward(); RoPE is intentionally not forwarded here.
    """
    def __init__(self, hidden_size, num_heads, *args, num_spatial_patches=64, **kwargs):
        super().__init__(hidden_size, num_heads, *args, **kwargs)
        self.num_spatial_patches = num_spatial_patches

    def forward(self, x, c, feat_rope=None):       # feat_rope intentionally ignored
        B, FS, D = x.shape
        S = self.num_spatial_patches
        F = FS // S

        # (B, F, S, D) → (B*S, F, D)
        x_t = x.reshape(B, F, S, D).permute(0, 2, 1, 3).reshape(B * S, F, D)
        c_t = c.unsqueeze(1).expand(-1, S, -1).reshape(B * S, D)

        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_t).chunk(6, dim=1)
        x_t = x_t + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x_t), shift_msa, scale_msa))   # no RoPE
        x_t = x_t.contiguous()
        x_t = x_t + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x_t), shift_mlp, scale_mlp))

        # (B*S, F, D) → (B, F*S, D)
        return x_t.reshape(B, S, F, D).permute(0, 2, 1, 3).reshape(B, FS, D)


# ═══════════════════════════════════════════════════════════════════════════════
#  DiT  (modified to support video)
# ═══════════════════════════════════════════════════════════════════════════════

class DiT(nn.Module):
    """
    Diffusion Transformer backbone.

    Supported input shapes
    ─────────────────────
    Image 1-D  (B, C, N)          num_frames=1 (default), int input_size
    Image 2-D  (B, C, H, W)       num_frames=1,           tuple input_size
    Video 1-D  (B, F, C, N)       num_frames>1,           int input_size
    Video 2-D  (B, F, C, H, W)    num_frames>1,           tuple input_size

    New parameters
    ─────────────
    num_frames (int, default 1)
        Number of video frames. Setting this to 1 gives the original image
        behaviour with no overhead.

    factorize (bool, default False)
        When True and num_frames > 1, the transformer stacks alternate between
        SpatialDiTBlock (attends within each frame) and TemporalDiTBlock
        (attends across frames). Attention cost is O(S² + F²) per layer
        instead of O((F·S)²), which matters for long videos.
        When False, standard DiTBlocks receive the full F·S token sequence.
    """
    def __init__(
        self,
        input_size=1024,
        num_frames=1,               # ← NEW
        patch_size=16,
        in_channels=1,
        cond_dim=2,
        class_dropout_prob=0.1,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_bias=True,
        use_swiglu=False,
        use_rope=False,
        attn_type="vanilla",
        window_size=64,
        qk_norm=False,
        num_experts=None,
        num_experts_per_tok=None,
        factorize=True,            # ← NEW
        **kwargs
    ):
        super().__init__()
        self.learn_sigma  = learn_sigma
        self.channels     = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size   = patch_size
        self.num_heads    = num_heads
        self.hidden_size  = hidden_size
        self.cond_dim     = cond_dim
        self.self_condition = False
        self.num_frames   = num_frames                          # ← NEW
        self.is_video     = (num_frames > 1)                   # ← NEW
        self.factorize    = factorize                          # ← NEW

        # ── Patch embedder & final layer ──────────────────────────────────────
        self.input_size = input_size
        if isinstance(input_size, int):
            self.is_1d = True
            if self.is_video:                                  # ← NEW branch
                self.x_embedder = PatchEmbed1DVideo(
                    input_size, num_frames, patch_size, in_channels, hidden_size)
            else:
                self.x_embedder = PatchEmbed1D(
                    input_size, patch_size, in_channels, hidden_size)
            self.final_layer = FinalLayer1D(hidden_size, patch_size,
                                            self.out_channels, bias=use_bias)
            print(f"Creating {'Video ' if self.is_video else ''}1D DiT")
        else:
            assert isinstance(input_size, tuple) and len(input_size) == 2
            self.is_1d = False
            if self.is_video:                                  # ← NEW branch
                self.x_embedder = PatchEmbed2DVideo(
                    input_size, num_frames, patch_size, in_channels, hidden_size)
            else:
                self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
            self.final_layer = FinalLayer(hidden_size, patch_size,
                                          self.out_channels, bias=use_bias)
            print(f"Creating {'Video ' if self.is_video else ''}2D DiT")

        # ── RoPE  (spatial only; unchanged logic) ─────────────────────────────
        if use_rope and self.is_1d:
            head_dim = hidden_size // num_heads
            # For video, spatial sequence length is still input_size // patch_size
            seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=head_dim, max_seq_len=seq_len)
        else:
            if use_rope and not self.is_1d:
                print("rope is only implemented for 1D DiT. Setting use_rope to False.")
            self.feat_rope = None

        # ── Condition / timestep embedders ────────────────────────────────────
        self.t_embedder = TimestepEmbedder(hidden_size, bias=use_bias)
        self.y_embedder = ConditionEmbedder(cond_dim, hidden_size, class_dropout_prob)

        # For video embedders, num_spatial_patches is per-frame; for image embedders
        # it equals num_patches (unchanged).
        num_spatial_patches = (self.x_embedder.num_spatial_patches  # ← NEW
                                if self.is_video
                                else self.x_embedder.num_patches)
        num_patches = self.x_embedder.num_patches
        print(f"Creating DiT with {num_patches} total patches "
              f"({'factorized' if (factorize and self.is_video) else 'full-attn'} mode).")

        # ── Transformer blocks ─────────────────────────────────────────────────
        if self.is_video and factorize:                        # ← NEW branch
            # Alternate: even indices → SpatialDiTBlock, odd → TemporalDiTBlock
            self.blocks = nn.ModuleList()
            for i in range(depth):
                if i % 2 == 0:
                    blk = SpatialDiTBlock(
                        hidden_size, num_heads,
                        num_frames=num_frames,
                        mlp_ratio=mlp_ratio, bias=use_bias,
                        use_swiglu=use_swiglu, attn_type=attn_type,
                        qk_norm=qk_norm, num_experts=num_experts,
                        num_experts_per_tok=num_experts_per_tok)
                else:
                    blk = TemporalDiTBlock(
                        hidden_size, num_heads,
                        num_spatial_patches=num_spatial_patches,
                        mlp_ratio=mlp_ratio, bias=use_bias,
                        use_swiglu=use_swiglu,
                        attn_type="vanilla",  # temporal always uses vanilla attn
                        qk_norm=qk_norm)
                self.blocks.append(blk)
        else:
            # Original block construction (full attention over all tokens)
            block_class = (partial(WindowBlock, window_size=window_size)
                           if attn_type == "window" else DiTBlock)
            self.blocks = nn.ModuleList([
                block_class(
                    hidden_size, num_heads,
                    mlp_ratio=mlp_ratio, bias=use_bias,
                    use_swiglu=use_swiglu, attn_type=attn_type,
                    qk_norm=qk_norm, num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok)
                for _ in range(depth)
            ])

        self.initialize_weights()

    # ── weight initialisation ─────────────────────────────────────────────────

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Spatial positional embedding: (1, S, D)
        # For video, S = patches-per-frame; for image, S = total patches (unchanged).
        num_sp = (self.x_embedder.num_spatial_patches   # ← MODIFIED
                  if self.is_video else self.x_embedder.num_patches)
        pos_embed = get_1d_sincos_pos_embed(self.hidden_size, num_sp)
        self.register_buffer("pos_embed",
                             torch.from_numpy(pos_embed).float().unsqueeze(0))  # (1, S, D)

        # Temporal positional embedding: (1, F, D)  [video only]            ← NEW
        if self.is_video:
            temp_embed = get_1d_sincos_pos_embed(self.hidden_size, self.num_frames)
            self.register_buffer("temporal_pos_embed",
                                 torch.from_numpy(temp_embed).float().unsqueeze(0))  # (1, F, D)

        # Patch projection init
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Condition / timestep MLPs
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN & output projections
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ── NEW: combined spatial + temporal positional embedding ─────────────────

    def _video_pos_embed(self):
        """
        Build (1, F*S, D) by adding broadcast spatial and temporal embeddings.

        Token layout:  [f0_s0 … f0_sS | f1_s0 … f1_sS | … | f(F-1)_s0 … f(F-1)_sS]

        spatial:   every frame gets the same (1, S, D) per-patch encoding
        temporal:  every spatial position within a frame gets the same (1, F, D)
                   frame-index encoding
        """
        S = self.x_embedder.num_spatial_patches
        F = self.num_frames
        D = self.hidden_size

        # (1, S, D) → (1, F, S, D) → (1, F*S, D)
        sp = self.pos_embed.unsqueeze(1).expand(-1, F, -1, -1).reshape(1, F * S, D)
        # (1, F, D) → (1, F, S, D) → (1, F*S, D)
        tp = self.temporal_pos_embed.unsqueeze(2).expand(-1, -1, S, -1).reshape(1, F * S, D)
        return sp + tp

    # ── unpatchify ────────────────────────────────────────────────────────────

    def unpatchify(self, x):                               # ← MODIFIED
        """
        Image 1-D:  (B, S,   p·C)    → (B, C, S·p)
        Image 2-D:  (B, S,   p²·C)   → (B, C, H, W)
        Video 1-D:  (B, F·S, p·C)    → (B, F, C, S·p)
        Video 2-D:  (B, F·S, p²·C)   → (B, F, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size

        if self.is_video:
            F = self.num_frames
            S = x.shape[1] // F          # spatial patches per frame

            if self.is_1d:
                x = x.reshape(x.shape[0], F, S, p, c)
                x = x.permute(0, 1, 4, 2, 3)              # (B, F, C, S, p)
                x = x.reshape(x.shape[0], F, c, S * p)    # (B, F, C, seq_len)
            else:
                h = self.input_size[0] // p
                w = self.input_size[1] // p
                # assert h * w == S, "Spatial patch count must form a square grid"
                x = x.reshape(x.shape[0], F, h, w, p, p, c)
                x = x.permute(0, 1, 6, 2, 4, 3, 5)        # (B, F, C, h, p, w, p)
                x = x.reshape(x.shape[0], F, c, h * p, w * p)   # (B, F, C, H, W)
        else:
            # Original image behaviour (unchanged)
            num_patches = x.shape[1]
            if self.is_1d:
                x = x.reshape(x.shape[0], num_patches, p, c)
                x = x.permute(0, 3, 1, 2)
                x = x.reshape(x.shape[0], c, num_patches * p)
            else:
                h = self.input_size[0]
                w = self.input_size[1]
                assert h * w == num_patches
                x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
                x = torch.einsum('nhwpqc->nchpwq', x)
                x = x.reshape(shape=(x.shape[0], c, h * p, h * p))

        return x

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x, t, classes, return_act=False, *args, **kwargs):
        """
        x : (B, C, N)        image 1-D
            (B, C, H, W)     image 2-D
            (B, F, C, N)     video 1-D
            (B, F, C, H, W)  video 2-D
        """
        if self.is_video:                                      # ← NEW branch
            x = self.x_embedder(x) + self._video_pos_embed()  # (B, F*S, D)
        else:
            x = self.x_embedder(x) + self.pos_embed           # (B, S, D)

        t  = self.t_embedder(t)
        force_drop_ids = kwargs.get("force_drop_ids", None)
        y  = self.y_embedder(classes, self.training, force_drop_ids)
        c  = t + y

        for block in self.blocks:
            x = block(x, c, self.feat_rope)

        act = x
        x   = self.final_layer(x, c)
        x   = self.unpatchify(x)

        if return_act:
            return x, act
        return x

    def forward_with_cond_scale(
        self,
        x,
        t,
        classes,
        cond_scale=6,
        rescaled_phi=0.7,
        remove_parallel_component=True,
        keep_parallel_frac=0,
        cfg_interval_start=0,
        *args,
        **kwargs,
    ):
        batch_size = x.shape[0]
        combined   = torch.cat([x, x], dim=0)
        force_drop_ids = torch.cat([
            torch.zeros((batch_size,), dtype=torch.bool, device=x.device),
            torch.ones( (batch_size,), dtype=torch.bool, device=x.device),
        ], dim=0)
        y_combined = torch.cat([classes, classes], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        model_out  = self.forward(combined, t_combined, y_combined,
                                  force_drop_ids=force_drop_ids)

        # Channel split: image layout is (B, C, …) so split on dim 1;
        # video layout is (B, F, C, …) so split on dim 2.       ← MODIFIED
        ch   = self.channels
        cdim = 2 if self.is_video else 1
        eps  = model_out.narrow(cdim, 0, ch)
        rest = model_out.narrow(cdim, ch, self.out_channels - ch)

        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        update = cond_eps - uncond_eps

        if remove_parallel_component:
            parallel, orthog = project(update, cond_eps)
            update = orthog + parallel * keep_parallel_frac

        half_eps = cond_eps + update * (cond_scale - 1)

        if cfg_interval_start > 0:
            if t[0] < cfg_interval_start:
                half_eps = cond_eps

        if rescaled_phi != 0:
            std_fn = partial(torch.std, dim=tuple(range(1, half_eps.ndim)), keepdim=True)
            rescaled_logits = half_eps * (std_fn(cond_eps) / std_fn(half_eps))
            half_eps = rescaled_logits * rescaled_phi + half_eps * (1. - rescaled_phi)

        eps       = torch.cat([half_eps, uncond_eps], dim=0)
        eps_sigma = torch.cat([eps, rest], dim=cdim)
        return eps_sigma.chunk(2, dim=0)[0]

    def get_2d_params(self):
        return [p for p in self.parameters() if p.dim() == 2]

    def get_1d_params(self):
        return [p for p in self.parameters() if p.dim() != 2]


# ─── sincos positional embedding helpers (unchanged) ────────────────────────────

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    if isinstance(grid_size, int):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
    else:
        grid_h = np.arange(grid_size[0], dtype=np.float32)
        grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_h.shape[0], grid_w.shape[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_1d_sincos_pos_embed(embed_dim, length):
    pos = np.arange(length, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


# ─── model factory functions ─────────────────────────────────────────────────────
# All existing factories work for video too — just pass num_frames=F and optionally
# factorize=True.  Example: DiT_B_2(num_frames=16, factorize=True)

def DiT_XL_1(**kwargs): return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)
def DiT_XL_2(**kwargs): return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
def DiT_XL_4(**kwargs): return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)
def DiT_XL_8(**kwargs): return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_1(**kwargs):  return DiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)
def DiT_L_2(**kwargs):  return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)
def DiT_L_4(**kwargs):  return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)
def DiT_L_8(**kwargs):  return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_1(**kwargs):  return DiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)
def DiT_B_2(**kwargs):  return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)
def DiT_B_4(**kwargs):  return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)
def DiT_B_8(**kwargs):  return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_1(**kwargs):  return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)
def DiT_S_2(**kwargs):  return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)
def DiT_S_4(**kwargs):  return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)
def DiT_S_8(**kwargs):  return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_XS_1(**kwargs): return DiT(depth=8, hidden_size=256, patch_size=1, num_heads=4, **kwargs)
def DiT_XS_2(**kwargs): return DiT(depth=8, hidden_size=256, patch_size=2, num_heads=4, **kwargs)
def DiT_XS_4(**kwargs): return DiT(depth=8, hidden_size=256, patch_size=4, num_heads=4, **kwargs)
def DiT_XS_8(**kwargs): return DiT(depth=8, hidden_size=256, patch_size=8, num_heads=4, **kwargs)

def DiT_XXS_1(**kwargs):  return DiT(depth=6, hidden_size=128, patch_size=1, num_heads=4, **kwargs)
def DiT_XXS_2(**kwargs):  return DiT(depth=6, hidden_size=128, patch_size=2, num_heads=4, **kwargs)
def DiT_XXS_4(**kwargs):  return DiT(depth=6, hidden_size=128, patch_size=4, num_heads=4, **kwargs)
def DiT_XXS_8(**kwargs):  return DiT(depth=6, hidden_size=128, patch_size=8, num_heads=4, **kwargs)

def DiT_XXXS_1(**kwargs): return DiT(depth=4, hidden_size=128, patch_size=1, num_heads=4, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/1': DiT_XL_1,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/1':  DiT_L_1,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/1':  DiT_B_1,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/1':  DiT_S_1,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT-XS/1': DiT_XS_1,  'DiT-XS/2': DiT_XS_2,  'DiT-XS/4': DiT_XS_4,  'DiT-XS/8': DiT_XS_8,
    'DiT-XXS/1': DiT_XXS_1, 'DiT-XXS/2': DiT_XXS_2, 'DiT-XXS/4': DiT_XXS_4, 'DiT-XXS/8': DiT_XXS_8,
    'DiT-XXXS/1': DiT_XXXS_1,
}