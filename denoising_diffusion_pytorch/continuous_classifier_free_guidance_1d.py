import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity, schedule

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import MixedPrecisionPolicy
from ema_pytorch import EMA

from contextlib import contextmanager
from torch.distributed._composable.fsdp import FSDPModule

from tqdm.auto import tqdm

from denoising_diffusion_pytorch.version import __version__

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True  

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_variance'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

# small helper modules

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift, gate = scale_shift
            x = x * (scale + 1) + shift

        x = self.proj(x)
        x = self.act(x)
        if exists(scale_shift):
            x = x * gate
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            # (scale, shift, gate) for block1 (scale, shift, gate) for block2
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim * 2 + dim_out * 4)
            # nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None
        self.dim_in = dim
        self.dim_out = dim_out
        self.block1 = Block(dim, dim_out, dropout)
        self.block2 = Block(dim_out, dim_out, dropout)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):

        # scale_shift_gate_block = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            # scale_shift_gate = torch.split(cond_emb, [self.dim_out, self.dim_out], dim=1)
            # first_chunk, second_chunk = cond_emb.chunk(2, dim=1)
            scale_shift_gate_block = torch.split(cond_emb, [self.dim_in, self.dim_in, self.dim_out, self.dim_out, self.dim_out, self.dim_out], dim=1)

        h = self.block1(x, scale_shift=scale_shift_gate_block[:3] if scale_shift_gate_block is not None else None)

        h = self.block2(h, scale_shift=scale_shift_gate_block[3:] if scale_shift_gate_block is not None else None)
        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32, qknorm=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.qk_norm = qknorm
        self.q_norm = nn.RMSNorm(hidden_dim) if qknorm else identity
        self.k_norm = nn.RMSNorm(hidden_dim) if qknorm else identity

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        # qkv = self.to_qkv(x).chunk(3, dim = 1)
        # q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q = rearrange(q, 'b c n -> b n c') 
        k = rearrange(k, 'b c n -> b n c')
        v = rearrange(v, 'b c n -> b n c')
        # q, k, v = qkv.unbind(1)  # Each is (B, N, C)
        
        # Apply normalization BEFORE reshaping into heads
        dtype = q.dtype
        if self.qk_norm:
            q = self.q_norm(q.to(self.q_norm.weight.dtype)).to(dtype)
            k = self.k_norm(k.to(self.k_norm.weight.dtype)).to(dtype) # (B, N, C)
        
        # Now reshape into multi-head format
        q = rearrange(q, 'b n (h d) -> b h d n', h=self.heads)  # (B, h, d, N)
        k = rearrange(k, 'b n (h d) -> b h d n', h=self.heads)  # (B, h, d, N)
        v = rearrange(v, 'b n (h d) -> b h d n', h=self.heads)  # (B, h, d, N)


        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(Module):
    def __init__(self, dim, heads = 4, dim_head = 32, qknorm=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.qk_norm = qknorm
        self.q_norm = nn.RMSNorm(hidden_dim) if qknorm else identity
        self.k_norm = nn.RMSNorm(hidden_dim) if qknorm else identity

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        # qkv = self.to_qkv(x).chunk(3, dim = 1)
        # q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q = rearrange(q, 'b c n -> b n c') 
        k = rearrange(k, 'b c n -> b n c')
        v = rearrange(v, 'b c n -> b n c')
        dtype = q.dtype
        if self.qk_norm:
            q = self.q_norm(q.to(self.q_norm.weight.dtype)).to(dtype)
            k = self.k_norm(k.to(self.k_norm.weight.dtype)).to(dtype)
        # n y c ESTAN AL REVES AQUI, PERO FUNCIONA MEJOR ¿?
        q, k, v = map(lambda t: rearrange(t, 'b n (h c) -> b h c n', h = self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v) # this outputs (b, h, c, n)
        out = rearrange(out, 'b h c n -> b (h c) n')

        # replace this with flash attention
        # q = q * self.scale
        # sim = einsum('b h d i, b h d j -> b h i j', q, k)
        # attn = sim.softmax(dim = -1)
        # out = einsum('b h i j, b h d j -> b h i d', attn, v)
        # out = rearrange(out, 'b h n d -> b (h d) n')
        # print(out.shape)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Conv1d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x, context):
        b, c, n = x.shape
        # context shape: (B, seq_len, context_dim)
        
        q = self.to_q(x)  # (B, hidden_dim, n)
        k = self.to_k(context)  # (B, seq_len, hidden_dim)
        v = self.to_v(context)  # (B, seq_len, hidden_dim)
        
        # Rearrange for multi-head attention
        q = rearrange(q, 'b (h d) n -> b h n d', h=self.heads)  # (B, heads, n, dim_head)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.heads)  # (B, heads, seq_len, dim_head)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.heads)  # (B, heads, seq_len, dim_head)

        q = q * self.scale

        # Attention: query positions attend to context sequence
        sim = einsum('b h i d, b h j d -> b h i j', q, k)  # (B, heads, n, seq_len)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (B, heads, n, dim_head)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D(Module):
    def __init__(
        self,
        dim,
        cond_dim, # number of conditioning classes
        cond_drop_prob=0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        dropout = 0.,
        learn_sigma = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn=False,
        qknorm=False,
        cross_attn=False,
        self_condition = False,
    ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob
        # determine dimensions
        self.channels = channels
        self.cond_dim = cond_dim
        self.self_condition = self_condition
        self.learn_sigma = learn_sigma
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.null_classes_emb = nn.Parameter(torch.randn(cond_dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(cond_dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )
        self.cross_attn = cross_attn

        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, classes_emb_dim=classes_dim, dropout = dropout)
        if cross_attn:
            outter_attention = partial(CrossAttention, context_dim=classes_dim, dim_head=attn_dim_head, heads=attn_heads, qknorm=qknorm)
        elif full_attn:
            outter_attention = partial(Attention, dim_head=attn_dim_head, heads=attn_heads, qknorm=qknorm)
        else:
            outter_attention = partial(LinearAttention, dim_head=attn_dim_head, heads=attn_heads, qknorm=qknorm)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, outter_attention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads, qknorm=qknorm)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, outter_attention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learn_sigma else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv1d(init_dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        rescaled_phi = 0.,
        remove_parallel_component = True,
        keep_parallel_frac = 0.,
        cfg_interval_start=0.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        update = logits - null_logits

        if remove_parallel_component:
            parallel, orthog = project(update, logits)
            update = orthog + parallel * keep_parallel_frac

        scaled_logits = logits + update * (cond_scale - 1.)
        if cfg_interval_start > 0:
            timestep = args[1][0] # args[1] is t
            if timestep < cfg_interval_start:
                scaled_logits = logits
        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        return interpolated_rescaled_logits
    
    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y) # mask
        return model_out.chunk(2, dim=1)[0] if self.learn_sigma else model_out


    def forward(self, x, time, classes, x_self_cond = None, cond_drop_prob=None, **kwargs):

        batch, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes,
                null_classes_emb
            )
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        c = self.classes_mlp(classes)
        x = self.init_conv(x)
        r = x.clone()

        if time is not None:
            t = self.time_mlp(time)
        else:
            t = None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            attention_args = {"x": x}
            if self.cross_attn:
                attention_args["context"] = c
            x = attn(**attention_args)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)
        # adds pad to x (the fisrt tensor) so that the second tensor has the same length on the last dimension as the first one 
        maybe_pad = lambda x, h: F.pad(x, (0, h.shape[-1] - x.shape[-1])) if h.shape[-1] != x.shape[-1] else x

        for block1, block2, attn, upsample in self.ups:
            res_connection = h.pop()
            x = maybe_pad(x, res_connection) # check if x needs to be padded
            x = torch.cat((x, res_connection), dim = 1)
            x = block1(x, t, c)

            res_connection = h.pop()
            x = maybe_pad(x, res_connection)
            x = torch.cat((x, res_connection), dim = 1)
            x = block2(x, t, c)
            attention_args = {"x": x}
            if self.cross_attn:
                attention_args["context"] = c
            x = attn(**attention_args)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1D(Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        channels = None,
        channel_first = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
    ):
        super().__init__()
        self.model = model
        self.channels = default(channels, lambda: self.model.channels)
        self.cond_dim = self.model.cond_dim
        self.self_condition = self.model.self_condition
        self.channel_first = channel_first
        self.seq_index = -2 if not channel_first else -1

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, classes, x_self_cond=None, clip_x_start = False, cond_scale = 6., rescaled_phi = 0.7, rederive_pred_noise = False):
        # model_output_null will be useful if i want to add cfg++ (i think)
        model_output = self.model.forward_with_cond_scale(x, t, classes, x_self_cond=x_self_cond, cond_scale=cond_scale, rescaled_phi=rescaled_phi)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        learned_var = None
        if self.model.learn_sigma:
            preds, learned_var = model_output.chunk(2, dim=1) # 1 is the channel dimension
        else:
            preds = model_output
        if self.objective == 'pred_noise':
            pred_noise = preds
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            # No se que es esto pero parece que siempre es false
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = preds
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = preds
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start, learned_var)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, classes, cond_scale=cond_scale, rescaled_phi=rescaled_phi, x_self_cond=x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)
        if preds.pred_variance is not None:
            model_mean, _, _ = self.q_posterior(x_start=x_start, x_t=x, t=t)
            min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract(torch.log(self.betas), t, x.shape)
            # The predicted variance is [-1, 1] for [min_var, max_var].
            frac = (preds.pred_variance + 1) / 2
            # interpolation in log space between min and max log variance
            posterior_log_variance = frac * max_log + (1 - frac) * min_log
            posterior_variance = torch.exp(posterior_log_variance)
        else:
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, x_self_cond=None, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, classes=classes, x_self_cond=x_self_cond, cond_scale=cond_scale, rescaled_phi=rescaled_phi, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale=6., rescaled_phi=0.7, return_noise=False, return_all_steps=False):
        batch, device = shape[0], self.betas.device

        noise = torch.randn(shape, device=device)
        img = noise 
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, classes, self_cond, cond_scale, rescaled_phi)
            imgs.append(img)
        
        # transpose here is to keep the same format as in flow matchin sampling
        ret = img if not return_all_steps else torch.stack(imgs, dim = 1).transpose(1, 0)

        ret = self.unnormalize(ret)

        if not return_noise:
            return ret

        return ret, noise

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True, return_noise = False, return_all_steps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        noise = torch.randn(shape, device = device)
        img = noise
        imgs = [img]
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, self_cond, cond_scale=cond_scale, rescaled_phi=rescaled_phi, clip_x_start=True, rederive_pred_noise=True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            imgs.append(img)

        ret = img if not return_all_steps else torch.stack(imgs, dim=1).transpose(1, 0)

        ret = self.unnormalize(ret)

        if not return_noise:
            return ret

        return ret, noise

    @torch.no_grad()
    def sample(self, classes, cond_scale = 6., rescaled_phi = 0.7, return_noise=False, return_all_steps=False):
        batch_size, channels = classes.shape[0], self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        shape = (batch_size, channels, self.seq_length) if self.channel_first else (batch_size, self.seq_length, channels)
        return sample_fn(classes, shape, cond_scale=cond_scale, rescaled_phi=rescaled_phi, return_noise=return_noise, return_all_steps=return_all_steps)

    @torch.no_grad()
    def interpolate(self, x1, x2, classes, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, classes, self_cond=self_cond)

        return img

    @torch.no_grad()
    def img_to_img(self, reference_input, classes, t=None):
        b, *_, device = *reference_input.shape, reference_input.device
        t = default(t, self.num_timesteps - 1)

        t_batched = torch.full((b,), t, device = device)

        # add noise to reference input
        img = self.q_sample(reference_input, t=t_batched)

        # denoise with the new class
        for i in tqdm(reversed(range(0, t)), desc = 'img to img sample time step', total = t):
            img, x_start = self.p_sample(img, i, classes)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise = None, ):#return_reduced_loss = True):
        b = x_start.shape[0]
        n = x_start.shape[self.seq_index]

        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        # predict and take gradient step
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, classes).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(x, t, classes, x_self_cond=x_self_cond)

        if self.model.learn_sigma:
            model_out, learned_var = model_out.chunk(2, dim=1)
             # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            var_loss = self._vb_terms_bpd(
                model_pred_noise=model_out.detach(),
                model_var_pred=learned_var.detach(),
                x_start=x_start,
                x_t=x,
                t=t,
                clip_denoised=False,
            )
            # scale this as openai say on the paper --> https://arxiv.org/pdf/2102.09672 sec 3.1
            var_loss = var_loss # * 0.001

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')

        # if not return_reduced_loss:
        #     return loss * extract(self.loss_weight, t, loss.shape)

        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        # add the variance loss term if learning variance
        loss = loss.mean() + (var_loss.mean() if self.model.learn_sigma else 0.)
        return loss

    def _vb_terms_bpd(
            self, model_pred_noise, model_var_pred, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        x_start_pred = self.predict_start_from_noise(x_t, t, model_pred_noise)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
            x_start_pred.clamp_(-1., 1.)
        model_mean, _, _ = self.q_posterior(x_start=x_start_pred, x_t=x_t, t=t)
        min_log = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        max_log = extract(torch.log(self.betas), t, x_t.shape)
        # The predicted variance is [-1, 1] for [min_var, max_var].
        frac = (model_var_pred + 1) / 2
        # interpolation in log space between min and max log variance
        posterior_log_variance = frac * max_log + (1 - frac) * min_log
    
        kl = normal_kl(
            true_mean, true_log_variance_clipped, model_mean, posterior_log_variance
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * posterior_log_variance
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return output

    def forward(self, img, *args, **kwargs):
        b, n, device, seq_length, = img.shape[0], img.shape[self.seq_index], img.device, self.seq_length

        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# diffusion utils

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs



def print_profiler_summary(prof):
    averages = prof.key_averages()
    cuda_attr = 'self_device_time_total'

    print("\n=== Top CUDA ops ===")
    print(averages.table(sort_by=cuda_attr, row_limit=20))

    print("\n=== Communication ops (NCCL) ===")
    comm_ops = [e for e in averages if "nccl" in e.key.lower() or "allreduce" in e.key.lower()]
    total_cuda_time = sum(getattr(e, cuda_attr) for e in averages)
    total_comm_time = sum(getattr(e, cuda_attr) for e in comm_ops)

    for e in sorted(comm_ops, key=lambda x: getattr(x, cuda_attr), reverse=True):
        print(f"  {e.key:50s}  cuda_time: {getattr(e, cuda_attr)/1e3:.2f} ms")

    if total_cuda_time > 0:
        print(f"\nTotal CUDA time : {total_cuda_time/1e3:.2f} ms")
        print(f"Total comm time : {total_comm_time/1e3:.2f} ms  ({100*total_comm_time/total_cuda_time:.1f}%)")
        print(f"Compute time    : {(total_cuda_time-total_comm_time)/1e3:.2f} ms  ({100*(1-total_comm_time/total_cuda_time):.1f}%)")

    prof.export_chrome_trace("trace_rank.json")    


@contextmanager
def unsharded(model):
    fsdp_modules = [m for m in model.modules() if isinstance(m, FSDPModule)]
    for m in fsdp_modules:
        m.unshard(async_op=False)
    try:
        yield
    finally:
        for m in fsdp_modules:
            m.reshard()

# trainer class
class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'bf16',
        split_batches = True,
        max_grad_norm = None,
        use_cpu=False,
        dataset_test=None,
        eta_min_scheduler=None,
        use_muon=False,
        compile_model=False
    ):
        super().__init__()

        # accelerator
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        fsdp_plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            reshard_after_forward=True,  # HYBRID_SHARD equivalent in FSDP2
            mixed_precision_policy=MixedPrecisionPolicy(
                # param_dtype=torch.bfloat16,   # all-gather in bf16
                reduce_dtype=torch.float32,   # reduce-scatter in fp32
            ),
        )
        self.accelerator = Accelerator(
            mixed_precision = mixed_precision_type if amp else 'no',
            cpu=use_cpu,
            dataloader_config=DataLoaderConfiguration(split_batches=split_batches),
            gradient_accumulation_steps=gradient_accumulate_every,
            # fsdp_plugin=fsdp_plugin
        )
        # fsdp_plugin=FSDPPlugin(
        #     state_dict_type="sharded",
        #     mixed_precision_policy=MixedPrecision(
        #         param_dtype=torch.bfloat16,
        #         reduce_dtype=torch.float32,
        #         buffer_dtype=torch.bfloat16,
        #     )
        # )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
        if dataset_test is not None:
            self.dl_test = DataLoader(dataset_test, batch_size = train_batch_size, shuffle=False, pin_memory=True, num_workers=4)
        else:
            self.dl_test = None

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        # if use_muon:
        #     neural_net = diffusion_model.model
        #     muon_paramaters = list(neural_net.blocks.parameters()) + list(neural_net.t_embedder.parameters()) + list(neural_net.final_layer.parameters())
        #     adam_parameters = list(neural_net.x_embedder.parameters()) + list(neural_net.y_embedder.parameters())
        #     self.opts = [torch.optim.Muon(muon_paramaters, lr=train_lr, weight_decay=1e-3), AdamW(adam_parameters, lr=train_lr, betas=adam_betas, weight_decay=1e-4, fused=True)]
        # else:
        #     self.opts = [AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=1e-4, fused=True)]
        eps = 1e-6 if mixed_precision_type == 'fp16' or mixed_precision_type == 'bf16' else 1e-8
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=1e-2, fused=True, eps=eps)
        # cosine annealing lr scheduler
        self.use_lr_scheduler = eta_min_scheduler is not None
        if use_muon:
            print("lr scheduler is deactivated when using muon, at least until i figure out how to do that with multiple optimizers")
            self.use_lr_scheduler = False
            
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.train_num_steps, eta_min=eta_min_scheduler)
            self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)

        # for logging results in a folder periodically

        # if self.accelerator.is_main_process:
        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.device, dtype=torch.float32) # 

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.cond_dim = diffusion_model.cond_dim
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        if compile_model:
            print("Compiling model...")
            self.model = torch.compile(self.model) # mode="reduce-overhead"
            # self.model.neural_net = torch.compile(self.model.neural_net) # mode="reduce-overhead"
            print("Model compiled")

        self.loss_history = []
        self.test_loss_history = []

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone, model_state_dict):
        lr = self.opt.param_groups[0]['lr']

        data = {
            'step': self.step,
            'model': model_state_dict,  # Use the passed dictionary
            # 'opts': [opt.state_dict() for opt in self.opts],
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.use_lr_scheduler else None,
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__,
            'lr': lr,
            'loss_history': torch.tensor(self.loss_history),
            'test_loss_history': torch.tensor(self.test_loss_history)
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)
        try:
            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])
        except:
            state_dict = data['model']
            new_state_dict = {}
            for key, value in state_dict.items():
                # Remove "module." prefix if it exists (or contains it)
                new_key = key.replace("module.", "") if "module." in key else key
                new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)

        self.step = data['step']
        # self.opt.load_state_dict(data['opt'])
        # for i, opt in enumerate(self.opts):
        #     opt.load_state_dict(data["opts"][i])
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
            
        if exists(data['loss_history']):
            self.loss_history = data['loss_history'].tolist()

        if "test_loss_history" in data:
            self.test_loss_history = data['test_loss_history'].tolist()
        
        if self.use_lr_scheduler and "scheduler" in data and self.use_lr_scheduler:
            self.scheduler.load_state_dict(data['scheduler'])
        
        if "lr" in data:
            print(f"Setting loaded learning rate to {data['lr']}")
            for param_group in self.opt.param_groups:
                param_group['lr'] = data['lr']

    def train(self, do_profiling=False):
        accelerator = self.accelerator
        device = accelerator.device
        profiler = None
        PROFILE_START_STEP = 25
        PROFILE_ACTIVE_STEPS = 15 
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                if do_profiling and self.step == PROFILE_START_STEP and profiler is None:
                    profiler = profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        schedule=schedule(wait=0, warmup=1, active=PROFILE_ACTIVE_STEPS, repeat=1),
                        record_shapes=True,
                        with_stack=True,
                        on_trace_ready=print_profiler_summary,
                    )
                    profiler.__enter__()
                    if accelerator.is_main_process:
                        print(f"[Profiler] Started at step {self.step}")

                self.model.train()
                total_loss = 0.
                # for _ in range(self.gradient_accumulate_every):
                data = next(self.dl)#.to(device)
                with accelerator.accumulate(self.model):
                    sequence, classes = data[0].to(device), data[1].to(device)
                    with self.accelerator.autocast():
                        loss = self.model(sequence, classes=classes)
                        # loss = loss / self.gradient_accumulate_every
                        # total_loss += loss.item()

                    self.accelerator.backward(loss)
                    # accelerator.wait_for_everyone()
                    if accelerator.sync_gradients:
                        if self.max_grad_norm is not None:
                            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        # Step increments only on actual updates (when gradients are accumulated and ready to do backward)
                        self.step += 1

                    # for opt in self.opts:
                    #     opt.step()
                    #     opt.zero_grad()
                    self.opt.step()
                    self.opt.zero_grad()
                    if self.use_lr_scheduler:
                        self.scheduler.step()
                if profiler is not None:
                    profiler.step()
                    if self.step >= PROFILE_START_STEP + PROFILE_ACTIVE_STEPS + 1:  # +1 for warmup
                        profiler.__exit__(None, None, None)
                        profiler = None
                        if accelerator.is_main_process:
                            print("[Profiler] Done. Stopping training.")
                        break  # remove this if you want training to continue after profiling
                # accelerator.wait_for_everyone()
                if accelerator.sync_gradients:
                    with unsharded(self.model):
                        if self.accelerator.is_main_process:
                            self.ema.update()
                    if accelerator.is_main_process:
                        total_loss += loss.detach().float().mean().cpu().item()
                        self.loss_history.append(total_loss)
                        pbar.set_description(f'loss: {total_loss:.5f}')
                        pbar.update(1)
                        # with unsharded(self.model):
                        #     self.ema.update()
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        model_state_dict = accelerator.get_state_dict(self.model)
                        milestone = self.step // self.save_and_sample_every
                        if self.dl_test is not None:
                        #     all_samples_list = []
                        #     with torch.no_grad():
                        #         test_losses = []
                        #         for data in self.dl_test:
                        #             sequence, classes = data[0].to(device), data[1].to(device)
                        #             with accelerator.autocast():
                        #                 pred = self.ema.ema_model.sample(classes=classes)
                        #             all_samples_list.append(pred.float())
                        #             mse = ((pred - sequence) ** 2).mean()
                        #             test_losses.append(mse.cpu().numpy())

                        #     all_samples = torch.cat(all_samples_list, dim = 0)
                            samples, sequences = self.eval_model(self.dl_test.dataset, batch_size=self.batch_size)
                            if accelerator.is_main_process:
                                mse = ((samples - sequences) ** 2).mean()
                                test_losses = mse.cpu().item()
                                self.test_loss_history.append(test_losses)
                                torch.save(samples, str(self.results_folder / f'sample-{milestone}.pt'))

                        if accelerator.is_main_process:
                            self.save(milestone, model_state_dict)
                            self.save_loss_plot()
                            self.ema.ema_model.train()

                if accelerator.sync_gradients:
                    accelerator.wait_for_everyone()
 
        if accelerator.is_main_process:
            self.save_loss_plot()
            
        accelerator.print('training complete')
        # profiler.__exit__(None, None, None)
    
    def eval_model(self, dataset_test, batch_size=32, **sampling_kwargs):
        # Prepare models
        dl_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        
        # Accelerate handles moving the model to the correct device even on 1 GPU
        model, test_dataloader = self.accelerator.prepare(self.ema.ema_model, dl_test)
        model.eval()
       
        # FIX: Only broadcast if we are in a distributed setup (more than 1 process)
        if self.accelerator.num_processes > 1:
            with torch.no_grad():
                for param in model.parameters():
                    # Broadcast from rank 0 to all other ranks
                    torch.distributed.broadcast(param.data, src=0)
            
            # Wait for broadcast to complete
            self.accelerator.wait_for_everyone()
        
        all_preds = []
        all_seqs = []
        
        for data in tqdm(test_dataloader, disable=not self.accelerator.is_main_process):
            sequence, classes = data[0], data[1]
            with torch.inference_mode():    
                # with self.accelerator.autocast():
                pred = model.sample(classes=classes, **sampling_kwargs)
                
                # gather_for_metrics works automatically for both single and multi-GPU
                gathered_pred, sequence = self.accelerator.gather_for_metrics((pred, sequence))
        
            if self.accelerator.is_main_process:
                all_preds.append(gathered_pred.cpu())
                all_seqs.append(sequence.cpu())
                
            del pred
            del gathered_pred
            del classes
            del sequence
            
        if self.accelerator.is_main_process:
            return torch.cat(all_preds, dim=0), torch.cat(all_seqs, dim=0)
        return None, None
    
    def save_loss_plot(self):
        plt.figure()
        plt.plot(self.loss_history, label='Loss')
        if self.test_loss_history:
            test_x_values = list(range(self.save_and_sample_every, self.step+1, self.save_and_sample_every))
            # print(test_x_values, self.test_loss_history)
            plt.plot(test_x_values, self.test_loss_history, label='Test Loss')    
        # Compute moving average
        window_size = 100
        if len(self.loss_history) >= window_size:
            moving_avg = np.convolve(self.loss_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.loss_history)), moving_avg, label=f'Moving Avg ({window_size})')
        plt.yscale('log')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss Evolution')
        plt.legend()
        plt.savefig(self.results_folder / "loss_evolution.png", bbox_inches="tight", pad_inches=0)
        plt.close()
