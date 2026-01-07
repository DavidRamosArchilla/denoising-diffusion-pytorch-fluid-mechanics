from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from torch.nn.attention import SDPBackend

# constants

AttentionConfig = namedtuple('AttentionConfig', ['backends'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class
class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = True,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')

        if device_version > version.parse('8.0'):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig([SDPBackend.FLASH_ATTENTION])
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.nn.attention.sdpa_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out

# Attention with rope and rmsnorm. Borrowed from https://github.dev/hustvl/LightningDiT/blob/main/models/lightningdit.py
class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        proj_bias: bool = True,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # q, k = self.q_norm(q), self.k_norm(k)
        # this is done this way to avoid dtype mismatch when using fp16/bf16
        q = self.q_norm(q.to(self.q_norm.weight.dtype))
        k = self.k_norm(k.to(self.k_norm.weight.dtype))
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, proj_bias=True, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, 'dimension must be divisible by number of heads'
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.heads = num_heads
        # TODO: temporary left qkv_bias unused
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.proj = nn.Sequential(
            nn.Linear(dim, dim, bias=proj_bias),
            # nn.RMSNorm(dim),
        )

    def forward(self, x, rope=None):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 2)
        q, k, v = map(lambda t: rearrange(t, 'b n (c h)-> b h n c', h = self.heads), qkv)

        # q = q.softmax(dim = -1)
        # k = k.softmax(dim = -1)
        # use the relu approach https://export.arxiv.org/pdf/2410.10629
        q = F.relu(q)
        k = F.relu(k.transpose(-2, -1))
        q = q * self.scale        

        # context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        # context = k.transpose(-2, -1) @ v  # b h e d
        context = k @ v
        out = q @ context  # b h d n
        out = rearrange(out, 'b h n c -> b n (h c)', h = self.heads)
        return self.proj(out)

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)



def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        max_seq_len=1024, # Set this large enough for your data (e.g. 1024 or 4096)
        theta = 10000,
    ):
        super().__init__()
        
        # 1. Generate the frequencies (1D only)
        # inv_freq shape: (dim // 2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        # 2. Generate position indices: [0, 1, ..., max_seq_len-1]
        t = torch.arange(max_seq_len).float()
        
        # 3. Compute outer product: (max_seq_len, dim // 2)
        freqs = torch.outer(t, inv_freq)
        
        # 4. Repeat frequencies to match the specific "rotate_half" format
        # Your previous code used repeat(..., '... n -> ... (n r)', r=2)
        # This doubles the last dim so it matches the input shape
        freqs = repeat(freqs, 'n d -> n (d r)', r=2)
        
        # 5. Compute Sin and Cos
        freqs_cos = freqs.cos() # Shape (max_seq_len, dim)
        freqs_sin = freqs.sin() # Shape (max_seq_len, dim)

        # Register as buffers (so they are saved with state_dict but not trained)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print(f'======== RoPE 1D initialized with shape {self.freqs_cos.shape} ========')

    def forward(self, t):
        # t shape: (Batch, Heads, Seq_Len, Dim)
        seq_len = t.shape[-2]
        
        # Slice the cached frequencies to the current sequence length
        # Reshape to (1, 1, Seq_Len, Dim) for broadcasting
        cos = self.freqs_cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.freqs_sin[:seq_len].view(1, 1, seq_len, -1)
        
        return t * cos + rotate_half(t) * sin