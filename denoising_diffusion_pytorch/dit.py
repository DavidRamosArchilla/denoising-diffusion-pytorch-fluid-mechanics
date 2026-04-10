# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
# https://github.com/facebookresearch/DiT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import repeat, rearrange, pack, unpack
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp # Attention
from .attend import Attention, VisionRotaryEmbeddingFast, LinearAttention, WindowAttention, CrossAttention, LinearCrossAttention
from functools import partial
from .basic_modules import SwiGLUFFN
# from . import is_triton_module_available
# from megablocks.layers.moe import MoE
# from megablocks.layers.arguments import Arguments

# _triton_modules_available = False
# if is_triton_module_available():
try:
    from .fastlinear.modules import TritonLiteMLA
except:
    print("Triton modules not available. TritonLiteMLA and TritonMBConvPreGLU will not be usable.")
#     _triton_modules_available = True

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, cond_dim, hidden_size, dropout_prob):
        super().__init__()
        # self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # self.cond_dim = cond_dim
        self.null_classes_emb = nn.Parameter(torch.randn(cond_dim))
        self.dropout_prob = dropout_prob

    def token_drop(self, cond_variables, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        batch = cond_variables.shape[0]
        if force_drop_ids is None:
            drop_ids = torch.rand(cond_variables.shape[0], device=cond_variables.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
        cond_variables = torch.where(
            rearrange(drop_ids, "b -> b 1"), null_classes_emb, cond_variables
        )
        return cond_variables

    def forward(self, cond_variables, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond_variables = self.token_drop(cond_variables, force_drop_ids)
        embeddings = self.mlp(cond_variables)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, bias=True, use_swiglu=False, attn_type="vanilla", qk_norm=False, num_experts=None, num_experts_per_tok=None, **attn_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=bias) # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # attention_class = LinearAttention if linear_attn else Attention
        # self.attn = attention_class(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=bias) # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attn_type == "vanilla":
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        elif attn_type == "linear":
            self.attn = LinearAttention(hidden_size, num_heads=num_heads, qkv_bias=bias, proj_bias=bias, qk_norm=qk_norm, **attn_kwargs)
        elif attn_type == "triton_linear":
            # if not _triton_modules_available:
            #     raise ValueError(
            #         f"{attn_type} type is not available due to _triton_modules_available={_triton_modules_available}."
            #     )
            # linear self attention with triton kernel fusion
            # TODO: Here the num_heads set to 36 for tmp used
            self.attn = TritonLiteMLA(hidden_size, num_heads=num_heads, eps=1e-8)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if num_experts is not None and num_experts_per_tok is not None:
            self.mlp = SparseMoeBlock(embed_dim=hidden_size, mlp_ratio=mlp_ratio, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok) # SparseMoeBlock
            # self.mlp = self._create_megablocks_moe(
            #     hidden_size=hidden_size,
            #     mlp_hidden_dim=mlp_hidden_dim,
            #     num_experts=num_experts,
            #     num_experts_per_tok=num_experts_per_tok
            # )
            # self.is_megablocks = True
        elif use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim), bias=bias)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=bias)
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x.contiguous()
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        # MLP/MoE block
        # mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        # # wihtou this, cuda can explode
        # mlp_input = mlp_input.contiguous().to(torch.bfloat16)
        # print(mlp_input.dtype)
        # print(f"Expert weights dtype: {next(self.mlp.experts.parameters()).dtype}")
        # if self.is_megablocks:
        #     # Megablocks MoE expects different input format
        #     # It returns (output, aux_loss)
        #     mlp_output, aux_loss = self.mlp(mlp_input)
        #     # Store aux_loss for later use in training
        #     if self.training and hasattr(self, 'aux_loss'):
        #         self.aux_loss += aux_loss
        # else:
        #     mlp_output = self.mlp(mlp_input)
        
        # x = x + gate_mlp.unsqueeze(1) * mlp_output
        
        return x

    # def _create_megablocks_moe(self, hidden_size, mlp_hidden_dim, num_experts, num_experts_per_tok):
    #     """
    #     Create a Megablocks MoE layer.
        
    #     Megablocks uses a different API and requires specific arguments.
    #     """
    #     # Create megablocks arguments
    #     args = Arguments(
    #         hidden_size=hidden_size,
    #         ffn_hidden_size=mlp_hidden_dim,
    #         moe_num_experts=num_experts,
    #         moe_top_k=num_experts_per_tok,
    #         shared_expert=True,
    #         shared_expert_hidden_size=hidden_size,
    #         moe_capacity_factor=1.0,  # Adjust based on your needs
    #         moe_loss_weight=0.01,  # Equivalent to aux_loss_alpha
    #         device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu',
    #         mlp_impl="grouped", 
    #         bf16=True,
    #         fp16=False
    #         # moe_expert_model_parallelism=True
    #     )        
    #     return MoE(args).to(torch.bfloat16)


def window_partition(x, window_size):
    """
    Args:
        x: (B, N, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, N, C = x.shape
    # x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    x = x.view(B, N // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)
    return windows


class WindowBlock(DiTBlock):
    """
    Block for DiT with window attention.
    """
    def __init__(self, hidden_size, num_heads, window_size, shift_size=0, *args, **kwargs):
        super().__init__(hidden_size, num_heads, *args, **kwargs)
        # override attention with window attention
        self.attn = WindowAttention(hidden_size, window_size=window_size, num_heads=num_heads, qkv_bias=False)
        self.window_size = window_size
        # shift_size will be ignored for the moment

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x_windows = window_partition(x, self.window_size) 
        attn_windows = self.attn(x_windows).view(x.shape) # window reverse operation
        x = x + gate_msa.unsqueeze(1) * attn_windows
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=bias) # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                                     1D DiT                                    #
#################################################################################

class PatchEmbed1D(nn.Module):
    """1D sequence to Patch Embedding"""
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, S)
        x = self.proj(x)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
    
class FinalLayer1D(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=bias) # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

#################################################################################
#                                      MoE                                      #
#################################################################################
class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    
        # print(bsz, seq_len, h)    
        ### compute gating score
        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0) 
            # print(self.up_proj.weight.size(), self.down_proj.weight.size())
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=-1)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, embed_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2, pretraining_tp=1):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList([MoeMLP(hidden_size=embed_dim, intermediate_size=int(mlp_ratio * embed_dim), pretraining_tp=pretraining_tp) for i in range(num_experts)])
        self.gate = MoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        # hardcoded. This makes that the tokens go through shared experts always. This should be great
        self.n_shared_experts = 2
        
        if self.n_shared_experts is not None:
            intermediate_size =  embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size=embed_dim, intermediate_size=intermediate_size, pretraining_tp=pretraining_tp)
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states) 
        # print(topk_idx.tolist(), print(len(topk_idx.tolist()))) 
        # global selected_ids_list
        # selected_ids_list.append(topk_idx.tolist())

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts): 
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).float()
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
    

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            
            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
   

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=1024,        
        patch_size=16,       # Patch size along sequence
        in_channels=1,
        cond_dim=2,
        class_dropout_prob=0.1,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_bias=True, # this is to use muon
        use_swiglu=False,
        use_rope=False,
        attn_type="vanilla",
        window_size=64,
        qk_norm=False,
        num_experts=None,
        num_experts_per_tok=None,
        **kwargs
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.self_condition = False  # Not used in DiT, this is here for interface compatibility
        if isinstance(input_size, int):
            self.x_embedder = PatchEmbed1D(input_size, patch_size, in_channels, hidden_size)
            self.final_layer = FinalLayer1D(hidden_size, patch_size, self.out_channels, bias=use_bias)
            print("Creating 1D DiT")
        else:
            assert isinstance(input_size, tuple) and len(input_size) == 2
            self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, bias=use_bias)
            print("Creating 2D DiT")

        if use_rope:
            head_dim = hidden_size // num_heads
            seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=head_dim,
                max_seq_len=seq_len,
            )
        else:
            self.feat_rope = None

        self.t_embedder = TimestepEmbedder(hidden_size, bias=use_bias)
        self.y_embedder = ConditionEmbedder(cond_dim, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        print(f"Creating DiT with {num_patches} patches.")
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        block_class = partial(WindowBlock, window_size=window_size) if attn_type == "window" else DiTBlock
        self.blocks = nn.ModuleList(
            [
                block_class(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    bias=use_bias,
                    use_swiglu=use_swiglu,
                    attn_type=attn_type,
                    qk_norm=qk_norm,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                )
                for _ in range(depth)
            ]
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            self.x_embedder.num_patches
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    
    def unpatchify(self, x):
        """
        x: (B, num_patches, patch_size * C)
        output: (B, C, S)
        """
        c = self.out_channels
        p = self.patch_size
        num_patches = x.shape[1]
        # TODO: no se si esto ira para 2d. Efectivamente, no.
        x = x.reshape(x.shape[0], num_patches, p, c)  # (B, num_patches, patch_size, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, num_patches, patch_size)
        x = x.reshape(x.shape[0], c, num_patches * p)  # (B, C, S)
        return x

    def forward(self, x, t, classes, return_act=False, *args, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        force_drop_ids = kwargs.get("force_drop_ids", None)
        y = self.y_embedder(classes, self.training, force_drop_ids)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            # x = checkpoint(block, x, c, self.feat_rope, use_reentrant=False)
            x = block(x, c, self.feat_rope)                      # (N, T, D)
        act = x
        x = self.final_layer(x, c)               # (B, num_patches, patch_size * out_channels)
        x = self.unpatchify(x)                   # (B, out_channels, S)
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
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
        batch_size = x.shape[0]
        combined = torch.cat([x, x], dim=0)
        force_drop_ids = torch.cat(
            [
                torch.zeros((batch_size,), dtype=torch.bool, device=x.device),
                torch.ones((batch_size,), dtype=torch.bool, device=x.device),
            ],
            dim=0,
        )
        y_combined = torch.cat([classes, classes], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        model_out = self.forward(combined, t_combined, y_combined, force_drop_ids=force_drop_ids)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # separate noise predictions and from variance predictions if present
        eps, rest = model_out[:, :self.channels], model_out[:, self.channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # return half_eps, uncond_eps
        update = cond_eps - uncond_eps
        if remove_parallel_component:
            parallel, orthog = project(update, cond_eps)
            update = orthog + parallel * keep_parallel_frac
        # half_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
        half_eps = cond_eps + update * (cond_scale - 1)

        if cfg_interval_start > 0:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        if rescaled_phi != 0:
            std_fn = partial(torch.std, dim = tuple(range(1, half_eps.ndim)), keepdim = True)
            rescaled_logits = half_eps * (std_fn(cond_eps) / std_fn(half_eps))
            half_eps = rescaled_logits * rescaled_phi + half_eps * (1. - rescaled_phi)

        eps = torch.cat([half_eps, uncond_eps], dim=0)
        eps_sigma = torch.cat([eps, rest], dim=1)
        # # return cfg eps and unconditioned eps
        return eps_sigma.chunk(2, dim=0)[0]
        # if rescaled_phi == 0.:
        #     return scaled_logits, null_logits

        # std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        # rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        # interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        # return interpolated_rescaled_logits, null_logits

class DiTCoordPE(DiT):
    def __init__(self, coords, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coord_embedder = CoordEmbedder(self.pos_embed.shape[-1], coords.shape[-1], num_frequencies=32)
        # self.coords = coords
        self.register_buffer("coords", coords.unsqueeze(0)) # (1, N, C)

    def forward(self, x, t, classes, return_act=False, *args, **kwargs):
        x = self.x_embedder(x) + self.coord_embedder(self.coords)
        t = self.t_embedder(t)
        force_drop_ids = kwargs.get("force_drop_ids", None)
        y = self.y_embedder(classes, self.training, force_drop_ids)
        c = t + y
        for block in self.blocks:
            x = block(x, c, self.feat_rope)
        act = x
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if return_act:
            return x, act
        return x

class DiTBlockMultiShape(nn.Module):
    """
    In Sana implementation they use cross attn + self attn. TODO: consider this approach too
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, bias=True, use_swiglu=False, attn_type="vanilla", qk_norm=False, num_experts=None, num_experts_per_tok=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=bias)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=bias)
        if attn_type == "vanilla":
            self.self_attn = Attention(hidden_size, num_heads, proj_bias=bias, qkv_bias=bias, qk_norm=qk_norm)
            self.cross_attn = CrossAttention(hidden_size, num_heads, proj_bias=bias, qkv_bias=bias, qk_norm=qk_norm)
        elif attn_type == "linear":
            self.self_attn = LinearAttention(hidden_size, num_heads, proj_bias=bias, qkv_bias=bias, qk_norm=qk_norm)
            self.cross_attn = LinearCrossAttention(hidden_size, num_heads, proj_bias=bias, qkv_bias=bias, qk_norm=qk_norm)
        else:
            raise ValueError(f"Unsupported attention type for DiTMultiShape: {attn_type}")
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim), bias=bias)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=bias)
        )

    def forward(self, x, c, context, feat_rope=None, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.self_attn(x, rope=feat_rope, mask=mask)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + self.cross_attn(x, context, mask=mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTMultiShape(DiT):
    def __init__(self, context_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # input size here should me max_input_size, which is the max sequence length among all shapes. However, the input_size of this patcher is useless
        self.context_embedder = PatchEmbed1D(
            kwargs["input_size"], kwargs["patch_size"], context_channels, kwargs["hidden_size"]
        )
        self.coord_embedder = CoordEmbedder(kwargs.get("hidden_size", 512), context_channels, num_frequencies=32)
        self.blocks = nn.ModuleList(
            [
                DiTBlockMultiShape(
                    kwargs.get("hidden_size", 512),
                    kwargs["num_heads"],
                    mlp_ratio=kwargs.get("mlp_ratio", 4.0),
                    bias=kwargs.get("use_bias", True),
                    use_swiglu=kwargs.get("use_swiglu", False),
                    attn_type=kwargs.get("attn_type", "vanilla"),
                    qk_norm=kwargs.get("qk_norm", False),
                    num_experts=kwargs.get("num_experts", None),
                    num_experts_per_tok=kwargs.get("num_experts_per_tok", None),
                )
                for _ in range(kwargs["depth"])
            ]
        )
        self.initialize_weights()
        
    def forward(self, x, t, classes, context, mask=None, return_act=False, *args, **kwargs):
        coord_embed = self.coord_embedder(context.transpose(1, 2))
        x = self.x_embedder(x) + coord_embed #+ self.pos_embed coord_embed
        context = self.context_embedder(context) + coord_embed
        t = self.t_embedder(t)
        force_drop_ids = kwargs.get("force_drop_ids", None)
        y = self.y_embedder(classes, self.training, force_drop_ids)
        c = t + y
        for block in self.blocks:
            x = block(x, c, context, self.feat_rope, mask)
        act = x
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if return_act:
            return x, act
        return x    

    def forward_with_cond_scale(
        self,
        x,
        t,
        classes,
        context,
        mask=None,
        cond_scale=6,
        rescaled_phi=0.7,
        remove_parallel_component=True,
        keep_parallel_frac=0,
        cfg_interval_start=0,
        *args,
        **kwargs,
    ):
        batch_size = x.shape[0]
        combined = torch.cat([x, x], dim=0)
        force_drop_ids = torch.cat(
            [
                torch.zeros((batch_size,), dtype=torch.bool, device=x.device),
                torch.ones((batch_size,), dtype=torch.bool, device=x.device),
            ],
            dim=0,
        )
        y_combined = torch.cat([classes, classes], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        context_combined = torch.cat([context, context], dim=0)
        mask_combined = None
        if mask is not None:
            mask_combined = torch.cat([mask, mask], dim=0)
        model_out = self.forward(combined, t_combined, y_combined, context_combined, mask_combined, force_drop_ids=force_drop_ids)
        
        eps, rest = model_out[:, :self.channels], model_out[:, self.channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        update = cond_eps - uncond_eps
        if remove_parallel_component:
            parallel, orthog = project(update, cond_eps)
            update = orthog + parallel * keep_parallel_frac
        half_eps = cond_eps + update * (cond_scale - 1)

        if cfg_interval_start > 0:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        if rescaled_phi != 0:
            std_fn = partial(torch.std, dim = tuple(range(1, half_eps.ndim)), keepdim = True)
            rescaled_logits = half_eps * (std_fn(cond_eps) / std_fn(half_eps))
            half_eps = rescaled_logits * rescaled_phi + half_eps * (1. - rescaled_phi)

        eps = torch.cat([half_eps, uncond_eps], dim=0)
        eps_sigma = torch.cat([eps, rest], dim=1)
        # return cfg eps, but not the unconditioned eps 
        return eps_sigma.chunk(2, dim=0)[0]

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width, or tuple (height, width)
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
    else:
        grid_h = np.arange(grid_size[0], dtype=np.float32)
        grid_w = np.arange(grid_size[1], dtype=np.float32)
    
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h.shape[0], grid_w.shape[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    """
    length: int, number of positions
    return: (length, embed_dim)
    """
    pos = np.arange(length, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def fourier_encode(coords, num_frequencies=32):
    """
    coords: (N, C) where C is 2 or 3
    out: (N, C * 2 * num_frequencies)
    """
    freqs = 2.0 ** torch.arange(num_frequencies, device=coords.device)  # (F,)
    x = coords.unsqueeze(-1) * freqs * 2 * np.pi  # (N, C, F)
    enc = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (N, C, 2F)
    return enc.flatten(-2)  # (N, C * 2F)


class CoordEmbedder(nn.Module):
    def __init__(self, embed_dim, coord_dim=3, num_frequencies=32):
        super().__init__()
        in_dim = coord_dim * 2 * num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords):
        """
        coords: (B, N, C) where C is 2 or 3
        out: (B, N, embed_dim)
        """
        enc = fourier_encode(coords.flatten(0, 1))      # (B*N, C*2F)
        emb = self.mlp(enc)                              # (B*N, embed_dim)
        return emb.view(*coords.shape[:2], -1)           # (B, N, embed_dim)

#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_1(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_1(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_1(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_1(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_XS_1(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=1, num_heads=4, **kwargs)

def DiT_XS_2(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=2, num_heads=4, **kwargs)

def DiT_XS_4(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=4, num_heads=4, **kwargs)

def DiT_XS_8(**kwargs):
    return DiT(depth=8, hidden_size=256, patch_size=8, num_heads=4, **kwargs)

def DiT_XXS_1(**kwargs):
    return DiT(depth=6, hidden_size=128, patch_size=1, num_heads=4, **kwargs)

def DiT_XXS_2(**kwargs):
    return DiT(depth=6, hidden_size=128, patch_size=2, num_heads=4, **kwargs)

def DiT_XXS_4(**kwargs):
    return DiT(depth=6, hidden_size=128, patch_size=4, num_heads=4, **kwargs)

def DiT_XXS_8(**kwargs):
    return DiT(depth=6, hidden_size=128, patch_size=8, num_heads=4, **kwargs)

def DiT_XXXS_1(**kwargs):
    return DiT(depth=4, hidden_size=128, patch_size=1, num_heads=4, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/1': DiT_XL_1,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/1':  DiT_L_1,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/1':  DiT_B_1,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/1':  DiT_S_1,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT-XS/1': DiT_XS_1,  'DiT-XS/2': DiT_XS_2,  'DiT-XS/4': DiT_XS_4,  'DiT-XS/8': DiT_XS_8,
    'DiT-XXS/1': DiT_XXS_1, 'DiT-XXS/2': DiT_XXS_2, 'DiT-XXS/4': DiT_XXS_4, 'DiT-XXS/8': DiT_XXS_8,
    'DiT-XXXS/1': DiT_XXXS_1
}