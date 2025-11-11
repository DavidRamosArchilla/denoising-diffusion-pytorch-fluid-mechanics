from torch_geometric.nn import GraphUNet
from torch_geometric.loader import DataLoader as DataLoaderGeom 
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils.repeat import repeat as repeat_geom

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)

import torch
import torch.nn.functional as F
from torch import nn, autocast
from torch import Tensor
from torch import nn, einsum
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import numpy as np
import matplotlib.pyplot as plt

from accelerate import Accelerator, DataLoaderConfiguration

import os
import math 
from pathlib import Path
from functools import partial
from random import random
from collections import namedtuple
from typing import Callable, List, Union
import warnings

from einops import rearrange, reduce, repeat, pack, unpack

from tqdm.auto import tqdm
from ema_pytorch import EMA


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


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

def divisible_by(numer, denom):
    return (numer % denom) == 0

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

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

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

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# def broadcast_for_graph_data(tensor, graph_data):
#     # shift = repeat(shift, 'b c -> b c n', n = x.shape[0] // shift.shape[0])
#     # scale = rearrange(scale, 'b c n -> (b n) c')
#     batch_size = tensor.shape[0]
#     num_points = graph_data.shape[0] // batch_size 
#     if len(tensor.shape) == 1:
#         tensor = repeat(tensor, 'b -> b n', n=num_points)
#         tensor = rearrange(tensor, 'b n -> (b n)')
#     else:
#         tensor = repeat(tensor, 'b 1 -> b n', n=num_points)
#         tensor = rearrange(tensor, 'b n -> (b n) 1')
#     return tensor

class GraphAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.to_qkv = GCNConv(dim, hidden_dim * 3, bias = False, improved=True)
        self.to_out = GCNConv(hidden_dim, dim, 1)

    def forward(self, x, edge_index, edge_weight):
        # b, c, h, w = x.shape
        batch_size = x.shape[0]
        x = self.norm(x)
        x = rearrange(x, 'b c n -> (b n) c')
        qkv = self.to_qkv(x, edge_index, edge_weight)# .chunk(3, dim = 1)
        qkv = rearrange(qkv, '(b n) c -> b c n', b=batch_size)
        qkv = qkv.chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = rearrange(out, 'b c n -> (b n) c')
        out = self.to_out(out, edge_index, edge_weight)
        out = rearrange(out, '(b n) c -> b c n', b=batch_size)
        return out
    
    def reset_parameters(self):
        self.to_qkv.reset_parameters()
        self.to_out.reset_parameters()


class GraphBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, classes_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None
        self.proj = GCNConv(dim, dim_out, improved=True)
        self.norm = RMSNorm(dim)
        print(dim, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_weight, time_emb=None, class_emb=None):
        # x: (B*N_points, dim) --> (B*N_points, dim_out)
        # print("x shape forward block", x.shape)
        batch_size = x.shape[0]
        x = self.norm(x)
        x = rearrange(x, 'b c n -> (b n) c')
        x = self.proj(x, edge_index, edge_weight)
        x = rearrange(x, '(b n) c -> b c n', b=batch_size)
        # print(x.shape)
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            # print(time_emb.shape, class_emb.shape)
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            # print("cond_emb", cond_emb.shape)
            # (B, time_emb_dim + classes_emb_dim) --> (B, dim_out * 2)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            # print("cond_emb", cond_emb.shape)
            scale_shift = cond_emb.chunk(2, dim = 1)
            # (B, dim_out), (B, dim_out) --> (B*N_points, dim_out), (B*N_points, dim_out)
            scale, shift = scale_shift
            # scale = repeat(scale, 'b c -> b c n', n = x.shape[0] // scale.shape[0])
            # shift = repeat(shift, 'b c -> b c n', n = x.shape[0] // shift.shape[0])
            # scale = rearrange(scale, 'b c n -> (b n) c')
            # shift = rearrange(shift, 'b c n -> (b n) c')
            x = x * (scale + 1) + shift
        x = self.act(x)
        
        return x

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.proj.reset_parameters()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class ConditionedGraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`False`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        cond_dim: int = None,
        cond_drop_prob=0.0,
        dim_mults=(1, 2, 4, 8),
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = False,
        act: Union[str, Callable] = 'relu',
        attn_heads=4,
        attn_dim_head=32
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        self.init_conv = GCNConv(in_channels, dim, improved=True)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.pool_ratios = repeat_geom(pool_ratios, len(in_out))
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        self.channels = in_channels
        self.cond_dim = cond_dim
        if cond_dim is not None: 
            time_dim = dim * 4
            sinu_pos_emb = SinusoidalPosEmb(dim)

            # time embeddings
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
            # class embeddings
            self.null_classes_emb = nn.Parameter(torch.randn(cond_dim))

            classes_dim = dim * 4

            self.classes_mlp = nn.Sequential(
                nn.Linear(cond_dim, classes_dim),
                nn.GELU(),
                nn.Linear(classes_dim, classes_dim)
            )
        else:
            time_dim, classes_dim = None, None

        self.downs = nn.ModuleList([])
        self.pools = torch.nn.ModuleList()
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                GraphBlock(dim=dim_in, dim_out=dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                GraphAttention(dim_out, attn_heads, attn_dim_head)
            ]))
            # pooling occurs after the GraphBlock
            self.pools.append(TopKPooling(dim_out, self.pool_ratios[ind]))

        self.mid_block1 = GraphBlock(dim=dims[-1], dim_out=dims[-1], time_emb_dim=time_dim, classes_emb_dim=classes_dim)
        self.mid_attn = GraphAttention(dims[-1], attn_heads, attn_dim_head)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            in_channels = dim_out if sum_res else 2 * dim_out # dim_in + dim_out # 2 * dim_out #
            self.ups.append(nn.ModuleList([
                GraphBlock(dim=in_channels, dim_out=dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                GraphAttention(dim_in, attn_heads, attn_dim_head)
            ]))
        self.out_conv = GCNConv(dim, out_channels, improved=True)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.init_conv.reset_parameters()
        self.mid_block1.reset_parameters()
        self.out_conv.reset_parameters()
        for block, attn in self.downs:
            block.reset_parameters()
            attn.reset_parameters()
        for block, attn in self.ups:
            block.reset_parameters()
            attn.reset_parameters()


    def forward(self, x: Tensor, edge_index: Tensor,
                time=None, classes=None, cond_drop_prob=None, batch: OptTensor = None, ) -> Tensor:
        # TODO: implement cfg
        # if cond_drop_prob > 0:
        #     pass
        conditioning_args = {}
        if self.cond_dim is not None:
            c = self.classes_mlp(classes)
            t = self.time_mlp(time)
            conditioning_args["time_emb"] = t
            conditioning_args["class_emb"] = c
        batch_size = x.shape[0]
        num_nodes = x.shape[2]
        if batch is None:
            # batch = edge_index.new_zeros(x.size(0))
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        edge_weight = x.new_ones(edge_index.size(1))
        initial_edge_index = edge_index
        initial_edge_weight = edge_weight

        batch_size = x.shape[0]
        x = rearrange(x, 'b c n -> (b n) c')
        x = self.init_conv(x, edge_index, edge_weight)
        x = rearrange(x, '(b n) c -> b c n', b=batch_size)
        xs, edge_indices, edge_weights = [], [], []
        perms = []

        for i, (block, attn) in enumerate(self.downs):
            # X: (B, C_in, N) --> (B, C_out, N)
            # print(f"block {i} x shape", x.shape)
            x = block(x, edge_index, edge_weight, **conditioning_args)
            x = attn(x, edge_index, edge_weight)
            # x = self.act(x)
            xs.append(x)
            edge_indices.append(edge_index)
            edge_weights.append(edge_weight)
            # print(f"block {i} x shape", x.shape, edge_index.shape)
            x = rearrange(x, 'b c n -> (b n) c')
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0)) #  * x.size(2)
            # print(f"block {i} x shape", x.shape, edge_index.shape)
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i](
                x, edge_index, edge_weight, batch)
            x = rearrange(x, '(b n) c -> b c n', b=batch_size)
            # print(f"block {i} x shape", x.shape, edge_index.shape)
            perms.append(perm)

        x = self.mid_block1(x, edge_index, edge_weight, **conditioning_args)
        x = self.mid_attn(x, edge_index, edge_weight)

        for i, (block, attn) in enumerate(self.ups):
            res = xs.pop()
            edge_index = edge_indices.pop()
            edge_weight = edge_weights.pop()
            perm = perms.pop()

            # this is basically the inverse operation of pooling. Since the residual conection has more points, half of
            # those points are populated with x where the permutation mask says (i'm not sure yet what is that perm variable)
            x = rearrange(x, 'b c n -> (b n) c')
            res = rearrange(res, 'b c n -> (b n) c')
            # print(x.shape, res.shape, perm.shape)
            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = rearrange(x, '(b n) c -> b c n', b=batch_size)
            x = block(x, edge_index, edge_weight, **conditioning_args)
            x = attn(x, edge_index, edge_weight)
            # x = self.act(x)

        x = rearrange(x, 'b c n -> (b n) c')
        x = self.out_conv(x, initial_edge_index, initial_edge_weight)
        x = rearrange(x, '(b n) c -> b c n', b=batch_size)
        return x

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor,
                    num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight,
                                  size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


class GraphDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        num_mesh_points,
        default_mesh_connectivity, # maybe a bit ugly, but for now it will make it work
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        use_cfg_plus_plus = False # https://arxiv.org/pdf/2406.08070
    ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        # self.num_features --> always 1?? (out_dim/in_dim)
        # self.self_condition = self.model.self_condition

        self.cond_dim = self.model.cond_dim
        # if isinstance(image_size, int):
        #     image_size = (image_size, image_size)
        # assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        # self.image_size = image_size
        self.num_mesh_points = num_mesh_points
        self.default_mesh_connectivity = default_mesh_connectivity

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # use cfg++ when ddim sampling

        self.use_cfg_plus_plus = use_cfg_plus_plus

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

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        return self.betas.device

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

    def model_predictions(self, x, t, classes, cond_scale = 6., rescaled_phi = 0.7, clip_x_start = False):
        # TODO: implement cfg training. and if i feel like, cfg++ too
        # model_output, model_output_null = self.model.forward_with_cond_scale(x, t, classes, x_self_cond, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        # x = rearrange(x, 'b c n -> (b n) c')
        model_output = self.model(x, self.default_mesh_connectivity, t, classes)
        # batch_size = t.shape[0]
        # model_output = rearrange(model_output, '(b n) c -> b c n', b=batch_size)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output # if not self.use_cfg_plus_plus else model_output_null

            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            x_start_for_pred_noise = x_start # if not self.use_cfg_plus_plus else maybe_clip(model_output_null)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)

            x_start_for_pred_noise = x_start
            # if self.use_cfg_plus_plus:
            #     x_start_for_pred_noise = self.predict_start_from_v(x, t, model_output_null)
            #     x_start_for_pred_noise = maybe_clip(x_start_for_pred_noise)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        batch_size = classes.shape[0]
        batched_times = torch.full((batch_size,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            classes=classes,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7):
        device = self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, num_inference_steps=None, clip_denoised = True):
        num_inference_steps = default(num_inference_steps, self.sampling_timesteps)
        batch, device, total_timesteps, sampling_timesteps, eta = classes.shape[0], self.betas.device, self.num_timesteps, num_inference_steps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, cond_scale=6., rescaled_phi=0.7, 
           sampler='', num_inference_steps=None):
        """
        Sample from the diffusion model.
        
        Args:
            classes: Class conditioning
            cond_scale: Classifier-free guidance scale
            rescaled_phi: Rescaling parameter
            sampler: One of ['ddpm', 'ddim', 'dpm-solver++']
            num_inference_steps: Number of steps (None uses defaults)
        """
        batch_size = classes.shape[0]
        num_mesh_points = self.num_mesh_points
        # shape = (batch_size, channels, self.image_size[0], self.image_size[1])
        shape = (batch_size * num_mesh_points, self.model.in_channels)
        if sampler == '':
            sampler = "ddim" if self.is_ddim_sampling else "ddpm"

        args = [classes, shape, cond_scale, rescaled_phi]
        if num_inference_steps is not None:
            args.append(num_inference_steps)

        if sampler == 'ddpm':
            return self.p_sample_loop(*args)
        elif sampler == 'ddim':
            # Use self.sampling_timesteps if num_inference_steps not specified
            return self.ddim_sample(*args)
        elif sampler == 'dpm-solver++':
            raise NotImplementedError("This is not implemented yet")
            # return self.dpm_solver_sample(*args)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    @torch.no_grad()
    def interpolate(self, x1, x2, classes, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        x_start = None
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, classes, self_cond)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # by default this was 0, the rearrange would need to be modified for graphunet
        # if self.offset_noise_strength > 0.:
        #     offset_noise = torch.randn(x_start.shape[:2], device = self.device)
        #     noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')
        # print(broadcast_for_graph_data(extract(self.sqrt_alphas_cumprod, t, x_start.shape), x_start).shape)
        # aqui falla, puede ser buena idea implementarme una funcion para hacer yo el 
        # broradcasting ya que al usar datos de grafos esto no puede pasar de manera automatica
        # shift = repeat(shift, 'b c -> b c n', n = x.shape[0] // shift.shape[0])
        # scale = rearrange(scale, 'b c n -> (b n) c')
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, edge_index, t, *, classes, noise = None):
        # b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, t, classes).pred_x_start
        #         x_self_cond.detach_()
        # x = rearrange(x, 'b c n -> (b n) c')
        model_out = self.model(x, edge_index, t, classes)#, x_self_cond)
        # batch_size = t.shape[0]
        # model_out = rearrange(model_out, '(b n) c -> b c n', b=batch_size)

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
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, data, *args, **kwargs):
        x = data.x
        edge_index = data.edge_index
        device = x.device
        b = data.num_graphs
        # assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = rearrange(x, '(b n) c -> b c n', b=b)
        # print(x.min(), x.max())
        x = normalize_to_neg_one_to_one(x)
        # print(x.min(), x.max())
        return self.p_losses(x, edge_index, t, *args, **kwargs)


class Trainer:
    def __init__(
        self,
        diffusion_model,
        *,
        dataset=None,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        save_best_and_latest_only=False,
        use_cpu=False,
        dl_collate_fn=None
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision_type if amp else 'no',
            cpu=use_cpu,
            dataloader_config=DataLoaderConfiguration(split_batches=split_batches)
        )

        # model
        self.model = diffusion_model

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        # assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        if (train_batch_size * gradient_accumulate_every) < 16:
            warnings.warn(f"WARNING: Your effective batch size (train_batch_size x gradient_accumulate_every) is {train_batch_size * gradient_accumulate_every}, which is less than 16. It is recommended to use at least 16 or above.")

        self.train_num_steps = train_num_steps

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        self.ds = dataset

        # this is the dataloader from torch geometric
        if dl_collate_fn is None:
            dl = DataLoaderGeom(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4)
        else:
            dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=dl_collate_fn)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = AdamW(diffusion_model.parameters(), lr = train_lr, betas = adam_betas, weight_decay=1e-4)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.cond_dim = diffusion_model.cond_dim
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.save_best_and_latest_only = save_best_and_latest_only
        self.all_losses = []

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'loss_history': torch.tensor(self.all_losses)
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
        if exists(data['loss_history']):
            self.all_losses = data['loss_history'].tolist()

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)#.to(device)
                    graphs, classes = data[0].to(device), data[1].to(device)
                    # print(f'Graph batch num graphs: {graphs.num_graphs},', graphs.x.shape, classes.shape)
                    with self.accelerator.autocast():
                        loss = self.model(graphs, classes=classes)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.all_losses.append(total_loss)
                    pbar.set_description(f'loss: {total_loss:.4f}')
                    pbar.update(1)
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            random_classes = torch.rand((self.num_samples, self.cond_dim), device=device)
                            batches = torch.split(random_classes, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(classes=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0).cpu()
                        accelerator.print(all_images.shape)

                        # utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        # grid = utils.make_grid(all_images, nrow=int(math.sqrt(self.num_samples)))
                        fig, axes = plt.subplots(3, 3, figsize=(12,12))

                        for i, ax in enumerate(axes.flat):
                            if all_images.shape[1] == 2: # if the image has 2 channels, keep just the first one
                                image = all_images[i, 0].squeeze()
                            else:
                                image = all_images[i].squeeze()
                            im = ax.imshow(image.squeeze())#, vmin=np.min(simulations), vmax=np.max(simulations))#, cmap="jet")
                            ax.set_title(f"Alpha1={random_classes[i][0].item():.2f}, Alpha2={random_classes[i][1].item():.2f}")
                            plt.colorbar(im, ax=ax)
                            ax.axis('off')
                        plt.savefig(self.results_folder / f"colored_grid{milestone}.png", bbox_inches="tight", pad_inches=0)
                        plt.close()

                        self.save(milestone)
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                
        if accelerator.is_main_process:
            plt.figure()
            plt.plot(self.all_losses, label='Loss')
            # Compute moving average
            window_size = 100
            if len(self.all_losses) >= window_size:
                moving_avg = np.convolve(self.all_losses, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(self.all_losses)), moving_avg, label=f'Moving Avg ({window_size})')
            plt.yscale('log')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss (log scale)')
            plt.title('Training Loss Evolution')
            plt.legend()
            plt.savefig(self.results_folder / "loss_evolution.png", bbox_inches="tight", pad_inches=0)
            plt.close()
        accelerator.print('training complete')


class TransonicRAE(Dataset):
    def __init__(self, data_directory, target_field, num_points=None, coef_norm=None):
        super(TransonicRAE, self).__init__()
       
        self.graph_dataset = []
        self.coef_norm = coef_norm
        self.num_points = num_points

        print("Processing dataset...")
        self.process_data(data_directory, target_field)
        
    def process_data(self, data_directory, target_field):

        print('Loading raw data')
        db_random = np.load(os.path.join(data_directory, 'db_random.npy'), allow_pickle=True).item()
        # db_cyc = np.load(os.path.join(data_directory, 'db_cyc.npy'), allow_pickle=True).item()
        db = db_random 
        # Merge db_random and db_cyc
        # db = {key: np.concatenate((db_random[key], db_cyc[key]), axis=0) for key in ['Pressure','Xcoordinate','Ycoordinate','Vinf','Alpha','idx']}
        print('Raw data Loaded, normalizing data')
    

        self.coef_norm = {'mean_in': None, 'std_in': None,'mean_out': None, 'std_out': None, 'min': None, 'max': None} 
        mean_out = db[target_field].mean()
        std_out = db[target_field].std()
        db[target_field] = (db[target_field]- mean_out)/std_out
        self.coef_norm['mean_out'] = mean_out
        self.coef_norm['std_out'] = std_out

         # Normalize condition data (Vinf and Alpha)
        cond_data = np.stack([db['Alpha'], db['Vinf']/347], axis=1)  # Normalize Vinf by 347
        mean_in = cond_data.mean(axis=0)
        std_in = cond_data.std(axis=0)
        cond_data = (cond_data - mean_in) / std_in
        self.coef_norm['mean_in'] = mean_in
        self.coef_norm['std_in'] = std_in
        

        for idx in tqdm(range(len(db['idx']))):
            if db['Vinf'][idx] / 347 >= 0.2:
                X_coord = 2*(db['Xcoordinate'][idx] - db['Xcoordinate'][idx].min()) / (db['Xcoordinate'][idx].max() - db['Xcoordinate'][idx].min()) - 1
                Y_coord = 2*(db['Ycoordinate'][idx] - db['Ycoordinate'][idx].min()) / (db['Ycoordinate'][idx].max() - db['Ycoordinate'][idx].min()) - 1
                pos = torch.tensor(np.stack((X_coord, Y_coord), axis=1), dtype=torch.float)
                output = torch.tensor(db[target_field][idx], dtype=torch.float).unsqueeze(-1)
                cond = torch.tensor(cond_data[idx], dtype=torch.float)
                cond = cond.repeat(pos.shape[0], 1)  # Repeat cond to match pos size

                if self.num_points is not None and pos.shape[0] > self.num_points:
                    subsample_indices = np.random.choice(pos.shape[0], self.num_points, replace=False)
                    pos = pos[subsample_indices]
                    output = output[subsample_indices]
                    cond = cond[subsample_indices]
                    
                # Concatenate pos and cond for x
                x = torch.cat([pos, cond], dim=1)

                # Create edges using k-nearest neighbors or radius graph
                edge_index = knn_graph(pos, k=8, batch=None, loop=False)
                # OR: edge_index = radius_graph(pos, r=0.1, batch=None, loop=False)
                # self.graph_dataset.append(Data(x=x, pos=pos, y=output))#, edge_index=edge_index))

                self.graph_dataset.append(Data(x=x, pos=pos, y=output, edge_index=edge_index))

    
    
    def create_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        if seed is not None:
            np.random.seed(seed)
        
        total_samples = len(self.graph_dataset)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        train_end = int(train_ratio * total_samples)
        val_end = train_end + int(val_ratio * total_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        self.train_dataset = [self.graph_dataset[i] for i in train_indices]
        self.val_dataset = [self.graph_dataset[i] for i in val_indices]
        self.test_dataset = [self.graph_dataset[i] for i in test_indices]


    def __getitem__(self, index):

        return self.graph_dataset[index]

    def __len__(self):
        return len(self.graph_dataset)

def train(device, model, train_loader, optimizer):
    model.train()
    avg_loss_per_var = torch.zeros(4, device = device)
    avg_loss = 0
    iter = 0
    
    for data in train_loader:
        data_clone = data.clone()
        data_clone = data_clone.to(device)          
        optimizer.zero_grad()  
        # data.x shape --> (batch_size * num_points, num_features)
        # print(data.x.shape, type(data.x))
        # print(data.edge_index.shape)
        # out = model(data_clone.x, data_clone.edge_index)
        # print(data.x)
        # break
        
        # batch_size = data_clone.num_graphs
        # sample_classes = torch.randint(0, 2, (batch_size, model.cond_dim), device=device).float()
        # sample_timesteps = torch.randint(0, 1000, (batch_size,), device=device).float()
        # out = model(data_clone.x, data_clone.edge_index, sample_timesteps, sample_classes)
        out = model(data_clone.x, data_clone.edge_index)# , sample_timesteps, sample_classes)

        targets = data_clone.y
        loss_criterion = nn.MSELoss(reduction = 'none')
        loss_per_var = loss_criterion(out, targets).mean(dim = 0)
        total_loss = loss_per_var.mean()
        total_loss.backward()
        
        optimizer.step()
        # scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss

        iter += 1
        # print(f"Iter {iter}, loss: {total_loss.item():.4f}", end='\r')
    
    print(f"\nAvg loss: {avg_loss.item()/iter:.4f}")

    return avg_loss.cpu().data.numpy()/iter, avg_loss_per_var.cpu().data.numpy()/iter


if __name__ == '__main__':
    from torch_geometric.data import Data
    data_directory = 'data/aeronef/'
    save_directory = 'data/aeronef/'
    batch_size = 16
    target_field = 'Pressure'
    # Create dataset objects
    train_dataset_obj = TransonicRAE(data_directory, target_field, num_points=10000)
    
    idx=100
    # Extract x and y from the dataset
    graph = train_dataset_obj.graph_dataset[idx]
    print(type(graph))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphUNet(
        in_channels=4, out_channels=1, hidden_channels=64, depth=3
    )
    # import pyLOM.NN
    # model = pyLOM.NN.MLP(
    #     input_size=4, 
    #     output_size=1,
    #     hidden_size=128,
    #     n_layers=4,
    #     p_dropouts=0.,
    #     # device='cpu'
    # )
    # TODO: es importante que cuando entrene modelo de difusion los datos esten en rango [0,1]
    model = ConditionedGraphUNet(
        dim=64,
        in_channels=4,
        out_channels=1,
        # cond_dim=2,
        cond_drop_prob=0.0,
        dim_mults=(1, 2, 4),
        pool_ratios=0.5,
        sum_res=False,
        act=nn.SiLU(),
    )
    print(model)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    train_dataset_obj.create_splits(train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, seed=42)
    train_loader = DataLoader(train_dataset_obj.train_dataset, batch_size=batch_size, shuffle=True)
    for i in range(5000):
        print(f"Epoch {i+1}/5000", flush=True)
        train(device, model, train_loader, optimizer)
        # Inference and plotting
    model.eval()
    with torch.no_grad():
        # Get a test sample
        test_sample = train_dataset_obj.test_dataset[0].to(device)
        
        # Make prediction
        prediction = model(test_sample.x, test_sample.edge_index)
        # prediction = model(test_sample.x)
        # batch_size = test_sample.num_graphs
        # sample_classes = torch.randint(0, 2, (1, model.cond_dim), device=device).float()
        # sample_timesteps = torch.randint(0, 1000, (1,), device=device).float()
        # prediction = model(test_sample.x, test_sample.edge_index, sample_timesteps, sample_classes)
        
        # Move to CPU and convert to numpy
        pred_np = prediction.cpu().numpy().squeeze()
        true_np = test_sample.y.cpu().numpy().squeeze()
        pos_np = test_sample.pos.cpu().numpy()
        
        # Denormalize if needed
        pred_denorm = pred_np * train_dataset_obj.coef_norm['std_out'] + train_dataset_obj.coef_norm['mean_out']
        true_denorm = true_np * train_dataset_obj.coef_norm['std_out'] + train_dataset_obj.coef_norm['mean_out']

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Ground truth
    scatter1 = axes[0].scatter(pos_np[:, 0], pos_np[:, 1], c=true_denorm, cmap='jet', s=1)
    axes[0].set_title('Ground Truth Pressure')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(scatter1, ax=axes[0])

    # Plot 2: Prediction
    scatter2 = axes[1].scatter(pos_np[:, 0], pos_np[:, 1], c=pred_denorm, cmap='jet', s=1)
    axes[1].set_title('Predicted Pressure')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(scatter2, ax=axes[1])

    # Plot 3: Error
    error = np.abs(true_denorm - pred_denorm)
    scatter3 = axes[2].scatter(pos_np[:, 0], pos_np[:, 1], c=error, cmap='Reds', s=1)
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(scatter3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('pressure_comparison.png', dpi=150)
    plt.show()

    # Basic error metrics
    mae = error.mean()
    max_err = error.max()

    # MSE / RMSE
    sq_error = (true_denorm - pred_denorm) ** 2
    mse = sq_error.mean()
    rmse = np.sqrt(mse)

    # Percentiles for absolute error
    pctiles = [50, 75, 90, 95, 99]
    abs_percentiles = np.percentile(error, pctiles)

    # Percentiles for squared error (report as RMSE at percentile by sqrt)
    sq_percentiles = np.percentile(sq_error, pctiles)
    rmse_percentiles = np.sqrt(sq_percentiles)

    # Relative errors (avoid divide by zero)
    eps = 1e-12
    rel_error = error / (np.abs(true_denorm) + eps)
    rel_mean = rel_error.mean()
    rel_max = rel_error.max()
    rel_percentiles = np.percentile(rel_error, pctiles)

    # Normalized RMSE (by std and by range)
    true_std = np.std(true_denorm)
    true_range = (np.max(true_denorm) - np.min(true_denorm)) if np.max(true_denorm) != np.min(true_denorm) else eps
    nrmse_std = rmse / (true_std + eps)
    nrmse_range = rmse / (true_range + eps)

    # Print summary
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Max Absolute Error: {max_err:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root MSE (RMSE): {rmse:.6f}")
    print(f"NRMSE (by std): {nrmse_std:.6f}, NRMSE (by range): {nrmse_range:.6f}")
    print()
    print("Absolute error percentiles:")
    for p, v in zip(pctiles, abs_percentiles):
        print(f"  {p}th percentile abs error: {v:.6f}")
    print()
    print("RMSE at squared-error percentiles (sqrt of percentile MSE):")
    for p, v in zip(pctiles, rmse_percentiles):
        print(f"  {p}th percentile RMSE: {v:.6f}")
    print()
    print("Relative error (absolute / |truth|) stats:")
    print(f"  Mean relative error: {rel_mean:.6f}")
    print(f"  Max relative error: {rel_max:.6f}")
    for p, v in zip(pctiles, rel_percentiles):
        print(f"  {p}th percentile relative error: {v:.6f}")

    # keep the original prints for convenience
    print()
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Max Error: {max_err:.4f}")
    print()

    # Create the scatter plot
    # plt.figure()
    # print("afdsfadfa", graph.x[:, 0].numpy().shape)
    # print(next(iter(train_dataset_obj.train_dataset)).x.shape)
    # plt.scatter(graph.x[:, 0].numpy(), graph.x[:, 1].numpy(), c=graph.y[:, 0].numpy(), cmap='viridis', s=0.5)
    # plt.colorbar(label='y value')
    # plt.xlabel('x[:, 0]')
    # plt.ylabel('x[:, 1]')
    # plt.title(f'Scatter plot for dataset index {idx}')
    # plt.savefig(f'{save_directory}/scatter_plot_{idx}.png', dpi=300)
    # plt.close()
