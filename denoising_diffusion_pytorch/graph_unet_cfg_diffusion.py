from torch_geometric.nn import GraphUNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph, radius_graph
from typing import Callable, List, Union

import torch
from torch import Tensor

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
)
from torch_geometric.utils.repeat import repeat as repeat_geom

import torch
from torch import nn, autocast
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import AdamW
import numpy as np

import os
import math 
from pathlib import Path
from functools import partial
from random import random
from collections import namedtuple

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


class GraphBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, classes_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None
        self.proj = GCNConv(dim, dim_out, improved=True)
        # self.norm = RMSNorm(dim_out)
        print(dim, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_weight, time_emb=None, class_emb=None):
        # x: (B*N_points, dim) --> (B*N_points, dim_out)
        # print(x.shape)
        x = self.proj(x, edge_index, edge_weight)
        # print(x.shape)
        # x = self.norm(x)
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            # (B, time_emb_dim + classes_emb_dim) --> (B, dim_out * 2)
            cond_emb = self.mlp(cond_emb)
            scale_shift = cond_emb.chunk(2, dim = 1)
            # (B, dim_out), (B, dim_out) --> (B*N_points, dim_out), (B*N_points, dim_out)
            scale, shift = scale_shift
            scale = repeat(scale, 'b c -> b c n', n = x.shape[0] // scale.shape[0])
            shift = repeat(shift, 'b c -> b c n', n = x.shape[0] // shift.shape[0])
            scale = rearrange(scale, 'b c n -> (b n) c')
            shift = rearrange(shift, 'b c n -> (b n) c')
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
        cond_dim: int,
        cond_drop_prob=0.0,
        dim_mults=(1, 2, 4, 8),
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = False,
        act: Union[str, Callable] = 'relu',
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
        self.cond_dim = cond_dim
        self.null_classes_emb = nn.Parameter(torch.randn(cond_dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(cond_dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        self.downs = nn.ModuleList([])
        self.pools = torch.nn.ModuleList()
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # WARN: no estoy seguro de que aqui vaya dim_in
            self.pools.append(TopKPooling(dim_out, self.pool_ratios[ind]))
            self.downs.append(
                GraphBlock(dim=dim_in, dim_out=dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
                # GraphBlock(dim=dim_in, dim_out=dim_out, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
            )

        self.mid_block1 = GraphBlock(dim=dims[-1], dim_out=dims[-1], time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            in_channels = dim_out if sum_res else 2 * dim_out # dim_in + dim_out # 2 * dim_out #
            self.ups.append(
                GraphBlock(dim=in_channels, dim_out=dim_in, time_emb_dim=time_dim, classes_emb_dim=classes_dim),
            )
        self.out_conv = GCNConv(dim, out_channels, improved=True)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.init_conv.reset_parameters()
        self.mid_block1.reset_parameters()
        self.out_conv.reset_parameters()
        for block in self.downs:
            block.reset_parameters()
        for block in self.ups:
            block.reset_parameters()


    def forward(self, x: Tensor, edge_index: Tensor,
                time, classes, cond_drop_prob=None, batch: OptTensor = None, ) -> Tensor:
        # TODO: implement cfg
        # if cond_drop_prob > 0:
        #     pass

        c = self.classes_mlp(classes)
        t = self.time_mlp(time)


        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))
        initial_edge_index = edge_index
        initial_edge_weight = edge_weight
        x = self.init_conv(x, edge_index, edge_weight)
        # print("after init conv:", x.shape)
        # r = x.clone()
        # x = self.downs[0](x, edge_index, edge_weight)
        # x = self.act(x)

        # xs = [x]
        # edge_indices = [edge_index]
        # edge_weights = [edge_weight]
        xs, edge_indices, edge_weights = [], [], []
        perms = []

        for i in range(len(self.downs)):

            x = self.downs[i](x, edge_index, edge_weight, time_emb=t, class_emb=c)
            x = self.act(x)

            # if i < len(self.downs) - 1:
            xs += [x]
            edge_indices += [edge_index]
            edge_weights += [edge_weight]

            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i](
                x, edge_index, edge_weight, batch)
            perms += [perm]
        # print("downsampled x shape:", x.shape)
        # for i in perms:
        #     print("perm shape:", i.shape)
        # for i in xs:
        #     print("xs shape:", i.shape)
        x = self.mid_block1(x, edge_index, edge_weight, time_emb=t, class_emb=c)
        # print("downsampled x shape:", x.shape)
        for i in range(len(self.ups)):
            res = xs.pop()
            edge_index = edge_indices.pop()
            edge_weight = edge_weights.pop()
            perm = perms.pop()
            # print("res shape:", res.shape)
            # print("perm shape:", perm.shape)
            # print(x.shape)

            # this is basically the inverse operation of pooling. Since the residual conection has more points, half of those points are populated with x where the permutation mask says (i'm not sure yet what is that perm variable)
            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            # print(x.shape)
            x = self.ups[i](x, edge_index, edge_weight, time_emb=t, class_emb=c)
            x = self.act(x) # if i < self.depth - 1 else x
            # print(x.shape)

        x = self.out_conv(x, initial_edge_index, initial_edge_weight)
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

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={len(self.ups)}, pool_ratios={self.pool_ratios})')



class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        num_mesh_points,
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
        self.self_condition = self.model.self_condition

        self.cond_dim = self.model.cond_dim
        # if isinstance(image_size, int):
        #     image_size = (image_size, image_size)
        # assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        # self.image_size = image_size
        self.num_mesh_points = num_mesh_points

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

    def model_predictions(self, x, t, classes, x_self_cond=None, cond_scale = 6., rescaled_phi = 0.7, clip_x_start = False):
        model_output, model_output_null = self.model.forward_with_cond_scale(x, t, classes, x_self_cond, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output if not self.use_cfg_plus_plus else model_output_null

            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            x_start_for_pred_noise = x_start if not self.use_cfg_plus_plus else maybe_clip(model_output_null)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)

            x_start_for_pred_noise = x_start
            if self.use_cfg_plus_plus:
                x_start_for_pred_noise = self.predict_start_from_v(x, t, model_output_null)
                x_start_for_pred_noise = maybe_clip(x_start_for_pred_noise)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, x_self_cond=None, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, x_self_cond, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, x_self_cond=None, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        # b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, x_self_cond=x_self_cond, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, classes, self_cond, cond_scale, rescaled_phi)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, num_inference_steps=None, clip_denoised = True):
        num_inference_steps = default(num_inference_steps, self.sampling_timesteps)
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, num_inference_steps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, self_cond, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

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
        batch_size, channels = classes.shape[0], self.channels
        shape = (batch_size, channels, self.image_size[0], self.image_size[1])
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

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, classes).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(x, t, classes, x_self_cond)

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

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


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
        db_cyc = np.load(os.path.join(data_directory, 'db_cyc.npy'), allow_pickle=True).item()
        
        # Merge db_random and db_cyc
        db = {key: np.concatenate((db_random[key], db_cyc[key]), axis=0) for key in ['Pressure','Xcoordinate','Ycoordinate','Vinf','Alpha','idx']}
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
        batch_size = data_clone.num_graphs
        sample_classes = torch.randint(0, 2, (batch_size, model.cond_dim), device=device).float()
        sample_timesteps = torch.randint(0, 1000, (batch_size,), device=device).float()
        out = model(data_clone.x, data_clone.edge_index, sample_timesteps, sample_classes)
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
        print(f"Iter {iter}, loss: {total_loss.item():.4f}", end='\r')
    
    print(f"\nAvg loss: {avg_loss.item()/iter:.4f}")

    return avg_loss.cpu().data.numpy()/iter, avg_loss_per_var.cpu().data.numpy()/iter


if __name__ == '__main__':
    from torch_geometric.data import Data
    import matplotlib.pyplot as plt
    data_directory = 'data/aeronef/'
    save_directory = 'data/aeronef/'
    batch_size = 32
    target_field = 'Pressure'
    # Create dataset objects
    train_dataset_obj = TransonicRAE(data_directory, target_field, num_points=512)
    
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
    model = ConditionedGraphUNet(
        dim=64,
        in_channels=4,
        out_channels=1,
        cond_dim=2,
        cond_drop_prob=0.0,
        dim_mults=(1, 2, 4),
        pool_ratios=0.5,
        sum_res=False,
        act='relu',
    )
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    train_dataset_obj.create_splits(train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, seed=42)
    train_loader = DataLoader(train_dataset_obj.train_dataset, batch_size=batch_size, shuffle=True)
    for i in range(300):
        train(device, model, train_loader, optimizer)
        # Inference and plotting
    model.eval()
    with torch.no_grad():
        # Get a test sample
        test_sample = train_dataset_obj.test_dataset[0].to(device)
        
        # Make prediction
        # prediction = model(test_sample.x, test_sample.edge_index)
        prediction = model(test_sample.x)
        
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

    print(f"Mean Absolute Error: {error.mean():.4f}")
    print(f"Max Error: {error.max():.4f}")

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
