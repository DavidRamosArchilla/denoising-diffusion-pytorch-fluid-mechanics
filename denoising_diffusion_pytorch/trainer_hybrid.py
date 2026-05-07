"""
Trainer1D — Hybrid TP + DDP trainer (replaces Accelerate-based version).

Parallelism layout (example: 4 GPUs, tp_size=2):
  DeviceMesh[0, :] → [GPU0, GPU1]  — model replica 0  (TP pair)
  DeviceMesh[1, :] → [GPU2, GPU3]  — model replica 1  (TP pair)
                      └────────────── DDP all-reduce across replicas

Launch:
  torchrun --nproc_per_node=4 train.py
"""

import os
import math
import contextlib
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from .basic_modules import SwiGLUFFN
from .attend import Attention
from timm.models.vision_transformer import Mlp 
from ema_pytorch import EMA

__version__ = "0.1.0"


# ── Utilities ─────────────────────────────────────────────────────────────────

def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return math.isqrt(num) ** 2 == num


# ── DTensor checkpoint helpers ────────────────────────────────────────────────

def to_regular_state_dict(obj):
    """
    Recursively walk a state_dict and convert every DTensor to a plain CPU
    tensor via .full_tensor().

    IMPORTANT — this is a COLLECTIVE operation: all TP ranks must call it
    simultaneously because full_tensor() issues an all-gather under the hood.
    Only rank 0 needs to actually write the result to disk.
    """
    if isinstance(obj, dict):
        return {k: to_regular_state_dict(v) for k, v in obj.items()}
    if isinstance(obj, DTensor):
        return obj.full_tensor().cpu()
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    return obj


def load_dtensor_state_dict(model, flat_sd):
    """
    Load a plain-tensor state dict (e.g., read from a checkpoint file) into a
    model whose parameters may already be DTensors (from a previous TP setup).

    distribute_tensor() re-shards each full tensor according to the placement
    spec already present on the model's parameters.

    All ranks must call this; distribute_tensor() is collective.
    """
    current_sd = model.state_dict()
    new_sd = {}
    for key, val in flat_sd.items():
        if key not in current_sd:
            continue
        cur = current_sd[key]
        if isinstance(cur, DTensor):
            # Re-shard the full tensor to match the existing DTensor placement
            new_sd[key] = distribute_tensor(
                val.to(cur.device),
                cur.device_mesh,
                cur.placements,
            )
        else:
            new_sd[key] = val.to(cur.device) if isinstance(val, torch.Tensor) else val
    model.load_state_dict(new_sd, strict=(len(new_sd) == len(current_sd)))


# ── Tensor Parallelism ────────────────────────────────────────────────────────

def apply_tensor_parallel(diffusion_model, tp_mesh):
    """
    Shard the DiT blocks inside GaussianDiffusion1D across the TP group.

    *** CUSTOMIZE THIS to match your DiT's attribute names. ***

    Two variants are shown for attention and MLP; uncomment whichever fits.
    The Col + Row pairing (Megatron trick) gives you only ONE all-reduce per
    block pair instead of one per layer — see the earlier explanation.
    """
    dit = diffusion_model.neural_net  # GaussianDiffusion1D → inner DiT

    for block in dit.blocks:

        # ── Attention projections ────────────────────────────────────────────
        attn = getattr(block, 'attn', None) or getattr(block, 'self_attn', None)
        if attn is not None:

            if hasattr(attn, 'to_q'):          # x-transformers / most custom DiTs
                parallelize_module(attn, tp_mesh, {
                    'to_q':     ColwiseParallel(),
                    'to_k':     ColwiseParallel(),
                    'to_v':     ColwiseParallel(),
                    'to_out.0': RowwiseParallel(),
                })

            elif hasattr(attn, 'q_proj'):      # standard PyTorch-style DiTs
                parallelize_module(attn, tp_mesh, {
                    'q_proj':   ColwiseParallel(),
                    'k_proj':   ColwiseParallel(),
                    'v_proj':   ColwiseParallel(),
                    'out_proj': RowwiseParallel(),
                })

        # ── MLP / FFN ─────────────────────────────────────────────────────────
        mlp = block.mlp
        if isinstance(mlp, Mlp):
            # timm-style: fc1 [dim → mlp_dim], fc2 [mlp_dim → dim]
            parallelize_module(mlp, tp_mesh, {
                'fc1': ColwiseParallel(),
                'fc2': RowwiseParallel(),
            })

        elif isinstance(mlp, SwiGLUFFN):
            # Fill these in once you check SwiGLUFFN's attribute names.
            # Typical SwiGLU has two "up" projections + one "down":
            #   'w1': ColwiseParallel()   ← gate branch
            #   'w2': ColwiseParallel()   ← value branch
            #   'w3': RowwiseParallel()   ← down projection
            parallelize_module(mlp, tp_mesh, {
                'w12': ColwiseParallel(),
                'w3': RowwiseParallel(),
            })
            pass
    return diffusion_model

# ── Trainer ───────────────────────────────────────────────────────────────────

class TrainerHybrid:
    def __init__(
        self,
        diffusion_model,
        dataset: Dataset,
        *,
        # ── NEW: parallelism ──────────────────────────────────────────────────
        tp_size: int = 2,           # GPUs per model replica (tensor parallel)
                                    # dp_size = world_size // tp_size (automatic)
        # ── training ─────────────────────────────────────────────────────────
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-4,
        train_num_steps: int = 100_000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every: int = 1_000,
        num_samples: int = 25,
        results_folder: str = './results',
        amp: bool = False,
        mixed_precision_type: str = 'bf16',
        max_grad_norm = None,
        dataset_test = None,
        eta_min_scheduler = None,
        compile_model: bool = False,
    ):
        super().__init__()

        # ── 1. Process group & device ─────────────────────────────────────────
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self._device = torch.device(f"cuda:{local_rank}")

        world_size = dist.get_world_size()
        assert world_size % tp_size == 0, (
            f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
        )
        dp_size = world_size // tp_size

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ── 2. 2-D Device Mesh ────────────────────────────────────────────────
        #
        #  mesh_shape = (dp_size, tp_size), row-major layout:
        #
        #   mesh[0, :] = [GPU0, GPU1]  ← replica 0  (communicate via TP all-reduce)
        #   mesh[1, :] = [GPU2, GPU3]  ← replica 1  (communicate via TP all-reduce)
        #        ↑↑ DDP all-reduces gradients across these two "rows"
        #
        self.mesh    = init_device_mesh(
            "cuda",
            mesh_shape=(dp_size, tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        self.tp_mesh = self.mesh["tp"]
        self.dp_mesh = self.mesh["dp"]

        self.dp_rank     = self.dp_mesh.get_local_rank()
        self.tp_rank     = self.tp_mesh.get_local_rank()
        self.global_rank = dist.get_rank()
        self.is_main     = (self.global_rank == 0)   # dp_rank==0 AND tp_rank==0

        # ── 3. Mixed precision ────────────────────────────────────────────────
        self.amp = amp
        self.mixed_precision_type = mixed_precision_type
        if amp and mixed_precision_type == 'fp16':
            self.amp_dtype = torch.float16
            self.scaler    = torch.amp.GradScaler('cuda')
        elif amp:                               # bf16: stable, no scaler needed
            self.amp_dtype = torch.bfloat16
            self.scaler    = None
        else:
            self.amp_dtype = torch.float32
            self.scaler    = None

        # ── 4. Model: TP sharding first, DDP wrapping second ──────────────────
        #
        #  Order matters:
        #    a) Move model to device
        #    b) Apply TP  → parameters become DTensors (sharded across tp_mesh)
        #    c) Create EMA  → deep-copies the already-sharded params ✓
        #    d) Wrap with DDP → syncs gradients across dp_mesh
        #
        self.model    = diffusion_model.to(self._device)
        self.channels = diffusion_model.channels
        self.cond_dim = diffusion_model.cond_dim

        if tp_size > 1:
            self.model = apply_tensor_parallel(self.model, self.tp_mesh)

        # ── 5. EMA ────────────────────────────────────────────────────────────
        #
        #  Created after TP so ema_model deep-copies the TP-sharded parameters.
        #  Updated on ALL ranks: since DDP keeps every DP replica in sync, each
        #  replica computes the identical EMA blend — no cross-replica broadcast
        #  needed. Both TP peers within a replica hold their respective shard.
        #
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self._device, dtype=torch.float32)

        # ── 6. DDP across the DP dimension ────────────────────────────────────
        # dp_group   = self.dp_mesh.get_group()
        # self.model = DDP(
        #     self.model,
        #     device_mesh=self.dp_mesh,   # DTensor-aware — skips the broken broadcast
        #     broadcast_buffers=False,    # buffers (e.g. BN running stats) also may be
        #                                 # DTensors; safer to skip broadcasting them too
        # )
        from torch.distributed._composable.replicate import replicate
        self.model = replicate(self.model, device_mesh=self.dp_mesh)
        self._register_dp_grad_hooks()
        # if compile_model:
        #     # torch.compile + DTensor is experimental before PyTorch 2.4.
        #     # If you hit graph-break errors try mode="reduce-overhead", or skip.
        #     print("Compiling model…")
        #     self.model = torch.compile(self.model, mode="reduce-overhead")
        #     print("Model compiled.")
        
        if compile_model:
            print("Compiling model...")
            dit = self.model.neural_net  # GaussianDiffusion1D → DiT

            # Compile the embedders and final layer — these are replicated (no DTensors)
            dit.x_embedder = torch.compile(dit.x_embedder)
            dit.t_embedder = torch.compile(dit.t_embedder)
            dit.y_embedder = torch.compile(dit.y_embedder)
            dit.final_layer = torch.compile(dit.final_layer)

            # Each block has replicated parts (norms, adaLN) and sharded parts (attn, mlp).
            # Compile at the block level — torch.compile will graph-break at DTensor ops
            # and compile everything around them, which still gives you a speedup on the
            # replicated portions.
            for i, block in enumerate(dit.blocks):
                dit.blocks[i] = torch.compile(block)

            print("Model compiled.")

        # ── 7. Hyper-parameters ────────────────────────────────────────────────
        assert has_int_squareroot(num_samples)
        self.num_samples               = num_samples
        self.save_and_sample_every     = save_and_sample_every
        self.batch_size                = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm             = max_grad_norm
        self.train_num_steps           = train_num_steps

        # ── 8. DataLoader ─────────────────────────────────────────────────────
        #
        #  CRITICAL: the sampler splits data across dp_size replicas ONLY.
        #  Both TP peers within the same replica must see the SAME batch —
        #  they cooperate on one forward pass, so feeding them different data
        #  would break the TP all-reduce.
        #
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=self.dp_rank,
            shuffle=True,
        )
        dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )
        self.dl            = cycle(dl)
        self.train_sampler = train_sampler  # expose for set_epoch() if needed

        if dataset_test is not None:
            test_sampler = DistributedSampler(
                dataset_test,
                num_replicas=dp_size,
                rank=self.dp_rank,
                shuffle=False,
            )
            self.dl_test = DataLoader(
                dataset_test,
                batch_size=train_batch_size,
                sampler=test_sampler,
                pin_memory=True,
                num_workers=4,
            )
        else:
            self.dl_test = None

        # ── 9. Optimizer ──────────────────────────────────────────────────────
        #  NOTE: fused=True is intentionally omitted — fused AdamW may not
        #  support DTensor parameters in all PyTorch versions. Re-enable once
        #  your version confirms compatibility.
        eps      = 1e-6 if mixed_precision_type in ('fp16', 'bf16') else 1e-8
        self.opt = AdamW(
            self.model.parameters(),
            lr=train_lr,
            betas=adam_betas,
            weight_decay=1e-2,
            eps=eps,
        )

        # ── 10. LR scheduler ──────────────────────────────────────────────────
        self.use_lr_scheduler = eta_min_scheduler is not None
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                T_max=train_num_steps,
                eta_min=eta_min_scheduler,
            )

        # ── 11. Misc state ────────────────────────────────────────────────────
        self.results_folder = Path(results_folder)
        if self.is_main:
            self.results_folder.mkdir(exist_ok=True)

        self.step              = 0
        self.loss_history      = []
        self.test_loss_history = []

    def _register_dp_grad_hooks(self):
        dp_group = self.dp_mesh.get_group()
        dp_size  = dist.get_world_size(group=dp_group)
        self._dp_handles = []
        self._dp_sync    = True

        def make_hook(p):
            def hook(grad):
                if self._dp_sync:
                    # ✅ Divide BEFORE all-reduce, inside the hook where grad mutation is allowed.
                    # This is equivalent to DDP's default "mean" reduction.
                    grad.div_(dp_size)
                    handle = dist.all_reduce(grad, group=dp_group, async_op=True)
                    self._dp_handles.append(handle)  # just the handle, no grad ref needed
                return grad
            return hook

        self._dp_hooks = []
        for p in self.model.parameters():
            if p.requires_grad:
                h = p.register_post_accumulate_grad_hook(make_hook(p))
                self._dp_hooks.append(h)

    @property
    def device(self):
        return self._device

    def _autocast(self):
        if self.amp:
            return torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype)
        return contextlib.nullcontext()

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, milestone):
        """
        Collective — every rank must call this.
        to_regular_state_dict() issues full_tensor() (an all-gather) on each
        DTensor param; only rank 0 then writes the gathered result to disk.
        """
        # Unwrap DDP (.module) to reach the GaussianDiffusion1D with DTensor params
        # model_sd = to_regular_state_dict(self.model.module.state_dict())
        model_sd = to_regular_state_dict(self.model.state_dict())

        ema_sd   = to_regular_state_dict(self.ema.state_dict())

        if self.is_main:
            torch.save(
                {
                    'step':              self.step,
                    'model':             model_sd,
                    'opt':               self.opt.state_dict(),
                    'scheduler':         self.scheduler.state_dict() if self.use_lr_scheduler else None,
                    'ema':               ema_sd,
                    'scaler':            self.scaler.state_dict() if self.scaler else None,
                    'version':           __version__,
                    'lr':                self.opt.param_groups[0]['lr'],
                    'loss_history':      torch.tensor(self.loss_history),
                    'test_loss_history': torch.tensor(self.test_loss_history),
                },
                str(self.results_folder / f'model-{milestone}.pt'),
            )

    def load(self, milestone):
        """
        All ranks read the same checkpoint file.
        load_dtensor_state_dict() re-shards each plain tensor into a DTensor
        using the placement spec already on the model's parameters — collective.
        """
        data = torch.load(
            str(self.results_folder / f'model-{milestone}.pt'),
            map_location=self._device,
            weights_only=True,
        )

        # Re-shard weights across TP ranks
        # load_dtensor_state_dict(self.model.module, data['model'])
        load_dtensor_state_dict(self.model, data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        # EMA was saved as plain tensors (gathered); load directly on all ranks
        # so every TP peer has its correct EMA shard for future eval/updates.
        self.ema.load_state_dict(data['ema'])

        if self.scaler and data.get('scaler'):
            self.scaler.load_state_dict(data['scaler'])

        if data.get('loss_history') is not None:
            self.loss_history = data['loss_history'].tolist()
        if data.get('test_loss_history') is not None:
            self.test_loss_history = data['test_loss_history'].tolist()

        if self.use_lr_scheduler and data.get('scheduler'):
            self.scheduler.load_state_dict(data['scheduler'])

        if 'lr' in data:
            for pg in self.opt.param_groups:
                pg['lr'] = data['lr']
            print(f"[rank {self.global_rank}] Loaded lr: {data['lr']}")

        if 'version' in data:
            print(f"[rank {self.global_rank}] Loading from version {data['version']}")

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        model  = self.model
        device = self._device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.is_main,
        ) as pbar:

            while self.step < self.train_num_steps:
                model.train()
                total_loss = 0.0
                self.opt.zero_grad()

                for acc_step in range(self.gradient_accumulate_every):
                    sequence, classes = next(self.dl)
                    sequence = sequence.to(device)
                    classes  = classes.to(device)
                    # On intermediate accumulation steps, suppress DDP's gradient
                    # all-reduce — it fires only on the last step, saving comms.
                    is_last_acc = (acc_step == self.gradient_accumulate_every - 1)
                    self._dp_sync = is_last_acc 
                    # sync_ctx = (
                    #     contextlib.nullcontext() if is_last_acc
                    #     else model.no_sync()
                    # )
                    # sync_ctx = (
                    #     contextlib.nullcontext() if is_last_acc
                    #     else self._repl_handle.no_sync()
                    # )
                    with self._autocast():
                        loss = model(sequence, classes=classes)
                        loss = loss / self.gradient_accumulate_every

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    for handle in self._dp_handles:
                        handle.wait()
                    self._dp_handles.clear()
                    total_loss += loss.detach().float().item()

                # Gradient clipping (unscale first if using fp16 scaler)
                if self.max_grad_norm is not None:
                    if self.scaler:
                        self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )

                if self.scaler:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()

                if self.use_lr_scheduler:
                    self.scheduler.step()

                self.step += 1

                # EMA update on ALL ranks: DDP guarantees every DP replica has
                # identical weights, so each replica computes the same EMA blend.
                # Each TP peer updates its own weight shard independently.
                self.ema.update()

                if self.is_main:
                    self.loss_history.append(total_loss)
                    pbar.set_description(f'loss: {total_loss:.5f}')
                    pbar.update(1)

                # ── Periodic eval & checkpoint ─────────────────────────────────
                if self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every

                    if self.dl_test is not None:
                        samples, sequences = self.eval_model(self.dl_test)
                        if self.is_main and samples is not None:
                            mse = ((samples - sequences) ** 2).mean().item()
                            self.test_loss_history.append(mse)
                            torch.save(
                                samples,
                                str(self.results_folder / f'sample-{milestone}.pt'),
                            )

                    # Collective: all ranks call save() even though only rank 0 writes
                    self.save(milestone)

                    if self.is_main:
                        self.save_loss_plot()

        if self.is_main:
            self.save_loss_plot()

        dist.barrier()
        if self.is_main:
            print('Training complete.')
        dist.destroy_process_group()

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_model(self, dl_test, **sampling_kwargs):
        """
        Run the EMA model on each DP rank's shard of the test set, then
        all-gather predictions from all DP replicas onto rank 0.

        TP peers within a replica cooperate on each forward pass exactly as
        they do during training — no special handling needed.
        """
        self.ema.ema_model.eval()
        dp_group = self.dp_mesh.get_group()
        dp_size  = self.dp_mesh.size()

        all_preds = []
        all_seqs  = []

        for data in tqdm(dl_test, disable=not self.is_main):
            sequence = data[0].to(self._device)
            classes  = data[1].to(self._device)

            with self._autocast():
                pred = self.ema.ema_model.sample(classes=classes, **sampling_kwargs)

            # Gather predictions & ground-truth from all DP replicas onto rank 0
            gathered_pred = [torch.zeros_like(pred)     for _ in range(dp_size)]
            gathered_seq  = [torch.zeros_like(sequence) for _ in range(dp_size)]
            dist.all_gather(gathered_pred, pred,     group=dp_group)
            dist.all_gather(gathered_seq,  sequence, group=dp_group)

            if self.is_main:
                all_preds.append(torch.cat(gathered_pred, dim=0).cpu())
                all_seqs.append( torch.cat(gathered_seq,  dim=0).cpu())

        self.ema.ema_model.train()

        if self.is_main:
            return torch.cat(all_preds), torch.cat(all_seqs)
        return None, None

    # ── Plotting ──────────────────────────────────────────────────────────────

    def save_loss_plot(self):
        plt.figure()
        plt.plot(self.loss_history, label='Loss')

        if self.test_loss_history:
            test_x = list(range(
                self.save_and_sample_every,
                self.step + 1,
                self.save_and_sample_every,
            ))
            plt.plot(test_x, self.test_loss_history, label='Test Loss')

        window = 100
        if len(self.loss_history) >= window:
            ma = np.convolve(
                self.loss_history, np.ones(window) / window, mode='valid'
            )
            plt.plot(
                range(window - 1, len(self.loss_history)), ma,
                label=f'Moving Avg ({window})',
            )

        plt.yscale('log')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss Evolution')
        plt.legend()
        plt.savefig(
            self.results_folder / 'loss_evolution.png',
            bbox_inches='tight', pad_inches=0,
        )
        plt.close()


"""
Trainer with hybrid Data Parallelism × Context Parallelism.

Launch with torchrun:
    torchrun --nproc_per_node=8 train.py --cp_degree 4

Process group layout (example: 8 GPUs, CP=4, DP=2):
    CP groups  (ring attention):  [0,1,2,3]  [4,5,6,7]
    DP groups  (grad reduction):  [0,4]  [1,5]  [2,6]  [3,7]

Each GPU sees sequence shard of length  L // cp_size  and a different batch
from the DP peers.  DDP only reduces gradients across the DP column so that
CP peers don't double-count gradients.
"""

import os
from contextlib import nullcontext
from pathlib import Path
from itertools import cycle

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm

# ── optional: ema-pytorch  (pip install ema-pytorch) ──────────────────────────
try:
    from ema_pytorch import EMA
    _has_ema = True
except ImportError:
    _has_ema = False


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_distributed() -> tuple[int, int, int]:
    """Initialise the default process group and return (rank, world_size, local_rank)."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_process_groups(cp_degree: int):
    """
    Build CP and DP sub-groups for the current rank.

    Layout:
        Ranks are arranged in a (dp_degree × cp_degree) grid.
        Row i  → CP group i  : ranks [i*cp, (i+1)*cp)   — ring-attn peers
        Col j  → DP group j  : ranks [j, j+cp, j+2*cp …] — DDP peers

    Returns:
        cp_group  : ProcessGroup for ring attention
        dp_group  : ProcessGroup for DDP gradient reduction
        cp_rank   : this rank's index within its CP group  (0 … cp_degree-1)
        dp_rank   : this rank's index within its DP group  (0 … dp_degree-1)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % cp_degree == 0, (
        f"world_size ({world_size}) must be divisible by cp_degree ({cp_degree})"
    )
    dp_degree = world_size // cp_degree

    cp_group = None
    for i in range(dp_degree):
        # ranks that process the SAME batch but DIFFERENT sequence shards
        ranks = list(range(i * cp_degree, (i + 1) * cp_degree))
        grp = dist.new_group(ranks)
        if rank in ranks:
            cp_group = grp

    dp_group = None
    for j in range(cp_degree):
        # ranks that process DIFFERENT batches but the SAME sequence shard index
        ranks = list(range(j, world_size, cp_degree))
        grp = dist.new_group(ranks)
        if rank in ranks:
            dp_group = grp

    cp_rank = rank % cp_degree
    dp_rank = rank // cp_degree
    return cp_group, dp_group, cp_rank, dp_rank


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class TrainerCP:
    def __init__(
        self,
        diffusion_model,
        dataset: Dataset,
        *,
        cp_degree: int = 1,
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-4,
        train_num_steps: int = 100_000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple = (0.9, 0.99),
        save_and_sample_every: int = 1000,
        results_folder: str = "./results",
        mixed_precision_type: str = "bf16",
        max_grad_norm: float | None = None,
        dataset_test: Dataset | None = None,
        eta_min_scheduler: float | None = None,
        dataloader_collate_fn=None,
        compile_model: bool = False,
    ) -> None:
 
        # ── Distributed init ─────────────────────────────────────────────────
        dist.init_process_group(backend="nccl")
        self.rank       = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device     = torch.device(f"cuda:{self.local_rank}")
        self.is_main    = self.rank == 0
        torch.cuda.set_device(self.device)
 
        self.cp_degree = cp_degree
        self.dp_degree = self.world_size // cp_degree
        assert self.world_size % cp_degree == 0
 
        # ── Device mesh: shape (dp, cp) ──────────────────────────────────────
        # init_device_mesh arranges ranks row-major:
        #   mesh[dp_rank, cp_rank] = global_rank
        # This gives us clean submeshes for free.
        if cp_degree > 1:
            self.mesh    = init_device_mesh("cuda", (self.dp_degree, cp_degree),
                                            mesh_dim_names=("dp", "cp"))
            self.cp_mesh = self.mesh["cp"]   # 1D mesh — passed to context_parallel
            self.dp_mesh = self.mesh["dp"]   # 1D mesh — used to get the DP process group
            dp_group     = self.dp_mesh.get_group()
            self.cp_rank = self.cp_mesh.get_local_rank()
            self.dp_rank = self.dp_mesh.get_local_rank()
        else:
            # Pure DP — no CP mesh needed
            self.mesh    = init_device_mesh("cuda", (self.world_size,),
                                            mesh_dim_names=("dp",))
            self.cp_mesh = None
            self.dp_mesh = self.mesh["dp"]
            dp_group     = self.dp_mesh.get_group()
            self.cp_rank = 0
            self.dp_rank = self.rank
 
        # ── Model ────────────────────────────────────────────────────────────
        self.raw_model = diffusion_model.to(self.device)
 
        if cp_degree > 1:
            # Inject the CP mesh into every Attention block
            self.raw_model.neural_net.set_cp_group(self.cp_mesh, self.cp_rank, cp_degree)
 
        # DDP only reduces gradients within the DP sub-group
        self.model = DDP(
            self.raw_model,
            device_ids=[self.local_rank],
            process_group=dp_group,
            find_unused_parameters=True,
        )
 
        if compile_model:
            if self.is_main:
                print("Compiling model …")
            self.model = torch.compile(self.model)
 
        # ── DataLoader ───────────────────────────────────────────────────────
        # CP peers share the same batch, so the sampler partitions over dp_degree
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dp_degree,
            rank=self.dp_rank,
            shuffle=True,
            drop_last=True,
        )
        self.dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=dataloader_collate_fn,
        )
        self.dl_iter = cycle(self.dl)
 
        if dataset_test is not None:
            self.dl_test = DataLoader(dataset_test, batch_size=train_batch_size,
                                      shuffle=False, pin_memory=True, num_workers=4,
                                      collate_fn=dataloader_collate_fn)
        else:
            self.dl_test = None
 
        # ── Optimizer & scheduler ────────────────────────────────────────────
        self.use_amp  = mixed_precision_type in ("bf16", "fp16")
        self.amp_dtype = torch.bfloat16 if mixed_precision_type == "bf16" else torch.float16
        self.scaler   = torch.amp.GradScaler(enabled=(mixed_precision_type == "fp16"))
 
        eps = 1e-6 if self.use_amp else 1e-8
        self.opt = AdamW(self.model.parameters(), lr=train_lr,
                         betas=adam_betas, weight_decay=1e-2, eps=eps)
 
        self.use_lr_scheduler = eta_min_scheduler is not None
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=train_num_steps, eta_min=eta_min_scheduler)
 
        # ── EMA ──────────────────────────────────────────────────────────────
        if self.is_main:
            assert _has_ema, "pip install ema-pytorch"
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device, dtype=torch.float32)
 
        # ── Misc ─────────────────────────────────────────────────────────────
        self.train_num_steps          = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm            = max_grad_norm
        self.save_and_sample_every    = save_and_sample_every
        self.batch_size               = train_batch_size
        self.results_folder           = Path(results_folder)
        if self.is_main:
            self.results_folder.mkdir(exist_ok=True)
        self.step         = 0
        self.loss_history: list[float] = []
 
    # ── Helpers ───────────────────────────────────────────────────────────────
 
    def _autocast(self):
        return (torch.autocast(device_type="cuda", dtype=self.amp_dtype)
                if self.use_amp else nullcontext())
 
    def _sync_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Average loss across CP ranks so the logged value is consistent."""
        if self.cp_degree == 1:
            return loss
        dist.all_reduce(loss, op=dist.ReduceOp.AVG,
                        group=self.cp_mesh.get_group())
        return loss
 
    # ── Training loop ─────────────────────────────────────────────────────────
 
    def train(self) -> None:
        pbar = tqdm(initial=self.step, total=self.train_num_steps,
                    disable=not self.is_main)
 
        while self.step < self.train_num_steps:
            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            total_loss = 0.0
 
            for _ in range(self.gradient_accumulate_every):
                data = next(self.dl_iter)
 
                if len(data) == 2:
                    sequence, classes = data[0].to(self.device), data[1].to(self.device)
                    model_inputs = {"classes": classes}
                elif len(data) == 3:
                    sequence, context, mask = (d.to(self.device) for d in data)
                    model_inputs = {"context": context, "mask": mask}
                elif len(data) == 4:
                    sequence, classes, context, mask = (d.to(self.device) for d in data)
                    model_inputs = {"classes": classes, "context": context, "mask": mask}
                else:
                    raise ValueError(f"Unexpected data fields: {len(data)}")
 
                # NOTE: no sequence sharding here — the model does it internally
                # (after the patcher, before transformer blocks) via _shard_sequence()
                with self._autocast():
                    loss = self.model(sequence, **model_inputs)
                    loss = loss / self.gradient_accumulate_every
 
                loss_for_log = self._sync_loss(loss.detach().clone())
 
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
 
                total_loss += loss_for_log.item()
 
            # ── Gradient clipping & optimizer step ───────────────────────────
            if self.max_grad_norm is not None:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
 
            if self.scaler.is_enabled():
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
 
            if self.use_lr_scheduler:
                self.scheduler.step()
 
            self.step += 1
 
            if self.is_main:
                self.ema.update()
                self.loss_history.append(total_loss)
                pbar.set_description(f"loss: {total_loss:.5f}")
                pbar.update(1)
 
            if self.step % self.save_and_sample_every == 0:
                dist.barrier()
                if self.is_main:
                    self.save(self.step // self.save_and_sample_every)
 
        pbar.close()
        dist.barrier()
        if self.is_main:
            print("Training complete.")
 
    # ── Checkpointing ─────────────────────────────────────────────────────────
 
    def save(self, milestone: int) -> None:
        if not self.is_main:
            return
        torch.save({
            "step": self.step,
            "model": self.model.module.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "loss_history": self.loss_history,
        }, self.results_folder / f"model-{milestone}.pt")
 
    def load(self, milestone: int) -> None:
        data = torch.load(self.results_folder / f"model-{milestone}.pt",
                          map_location=self.device)
        self.step = data["step"]
        self.model.module.load_state_dict(data["model"])
        self.opt.load_state_dict(data["opt"])
        if self.is_main:
            self.ema.load_state_dict(data["ema"])
        self.loss_history = data.get("loss_history", [])

# context parallel trainer    
class TrainerCPOld:
    def __init__(
        self,
        diffusion_model,
        dataset: Dataset,
        *,
        # ── parallelism ───────────────────────────────────────────────────────
        cp_size: int = 4,         # GPUs per context-parallel group
                                  # dp_size = world_size // cp_size (automatic)
        # ── training ──────────────────────────────────────────────────────────
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-4,
        train_num_steps: int = 100_000,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every: int = 1_000,
        num_samples: int = 25,
        results_folder: str = './results',
        amp: bool = False,
        mixed_precision_type: str = 'bf16',
        max_grad_norm = None,
        dataset_test = None,
        eta_min_scheduler = None,
        compile_model: bool = False,
    ):
        super().__init__()
 
        # ── 1. Process group & device ─────────────────────────────────────────
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self._device = torch.device(f"cuda:{local_rank}")
 
        world_size = dist.get_world_size()
        assert world_size % cp_size == 0, (
            f"world_size ({world_size}) must be divisible by cp_size ({cp_size})"
        )
        dp_size = world_size // cp_size
 
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
 
        # ── 2. 2-D Device Mesh ────────────────────────────────────────────────
        #
        #  mesh_shape = (dp_size, cp_size), row-major:
        #
        #    mesh[0, :] = [GPU0, ..., GPU_{cp-1}]  <- DP replica 0
        #                  all hold seq chunks of the SAME sample
        #    mesh[1, :] = [GPU_{cp}, ..., GPU_{2cp-1}]  <- DP replica 1
        #                  all hold seq chunks of a DIFFERENT sample
        #
        #  CP ranks communicate (K/V all-gather): fast intra-node NVLink.
        #  DP ranks communicate (grad all-reduce): once per optimizer step.
        #
        self.mesh    = init_device_mesh(
            "cuda",
            mesh_shape=(dp_size, cp_size),
            mesh_dim_names=("dp", "cp"),
        )
        self.cp_mesh = self.mesh["cp"]
        self.dp_mesh = self.mesh["dp"]
 
        self.cp_rank     = self.cp_mesh.get_local_rank()
        self.cp_size     = cp_size
        self.dp_rank     = self.dp_mesh.get_local_rank()
        self.dp_size     = dp_size
        self.global_rank = dist.get_rank()
        self.is_main     = (self.global_rank == 0)
 
        # ── 3. Mixed precision ────────────────────────────────────────────────
        self.amp = amp
        self.mixed_precision_type = mixed_precision_type
        if amp and mixed_precision_type == 'fp16':
            self.amp_dtype = torch.float16
            self.scaler    = torch.amp.GradScaler('cuda')
        elif amp:
            self.amp_dtype = torch.bfloat16
            self.scaler    = None
        else:
            self.amp_dtype = torch.float32
            self.scaler    = None
 
        # ── 4. Model ──────────────────────────────────────────────────────────
        # Unlike TP, weights are fully replicated across all ranks.
        # No parallelize_module / DTensor needed.
        self.model    = diffusion_model.to(self._device)
        self.channels = diffusion_model.channels
        self.cond_dim = diffusion_model.cond_dim
 
        # ── 5. Inject cp_group into every Attention block ─────────────────────
        # Attention.forward() reads self.cp_group to decide whether to
        # all-gather K/V. Set once at init so no per-step changes needed.
        cp_group = self.cp_mesh.get_group()
        self._inject_cp_group(diffusion_model, cp_group)
 
        # ── 6. EMA ────────────────────────────────────────────────────────────
        # EMA model also needs cp_group so eval sampling works correctly
        # (its attention layers will be called during sample()).
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.ema.to(self._device, dtype=torch.float32)
        self._inject_cp_group(self.ema.ema_model, cp_group)
 
        # ── 7. Gradient hooks for DP all-reduce ───────────────────────────────
        # Plain tensors (no DTensors), hooks registered directly on params.
        # div_ happens inside the hook during backward where it is permitted.
        # Outside backward, leaf variable grads are frozen and div_ would raise.
        self._dp_handles = []
        self._dp_sync    = True
        self._register_dp_grad_hooks()
 
        # if compile_model:
        #     # torch.compile works cleanly here — no DTensors in the graph.
        #     # The CP all-gather in Attention.forward() is compiled too.
        #     print("Compiling model...")
        #     self.model = torch.compile(self.model)
        #     print("Model compiled.")
        if compile_model:
            print("Compiling model...")
            dit = self.model.neural_net  # GaussianDiffusion1D → DiT

            # Compile the embedders and final layer — these are replicated (no DTensors)
            dit.x_embedder = torch.compile(dit.x_embedder)
            dit.t_embedder = torch.compile(dit.t_embedder)
            dit.y_embedder = torch.compile(dit.y_embedder)
            dit.final_layer = torch.compile(dit.final_layer)

            # Each block has replicated parts (norms, adaLN) and sharded parts (attn, mlp).
            # Compile at the block level — torch.compile will graph-break at DTensor ops
            # and compile everything around them, which still gives you a speedup on the
            # replicated portions.
            for i, block in enumerate(dit.blocks):
                dit.blocks[i] = torch.compile(block)

            print("Model compiled.")
 
        # ── 8. Hyper-parameters ───────────────────────────────────────────────
        assert has_int_squareroot(num_samples)
        self.num_samples               = num_samples
        self.save_and_sample_every     = save_and_sample_every
        self.batch_size                = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm             = max_grad_norm
        self.train_num_steps           = train_num_steps
 
        # ── 9. DataLoader ─────────────────────────────────────────────────────
        #
        #  CRITICAL: split data across dp_size replicas ONLY.
        #  All cp_size ranks within a replica process the SAME sample but each
        #  holds a different seq/cp_size chunk of it. The DataLoader must not
        #  know about CP — it just sees dp_size independent workers.
        #
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=self.dp_rank,
            shuffle=True,
        )
        dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
        )
        self.dl            = cycle(dl)
        self.train_sampler = train_sampler
 
        if dataset_test is not None:
            test_sampler = DistributedSampler(
                dataset_test,
                num_replicas=dp_size,
                rank=self.dp_rank,
                shuffle=False,
            )
            self.dl_test = DataLoader(
                dataset_test,
                batch_size=train_batch_size,
                sampler=test_sampler,
                pin_memory=True,
                num_workers=4,
            )
        else:
            self.dl_test = None
 
        # ── 10. Optimizer ─────────────────────────────────────────────────────
        eps      = 1e-6 if mixed_precision_type in ('fp16', 'bf16') else 1e-8
        self.opt = AdamW(
            self.model.parameters(),
            lr=train_lr,
            betas=adam_betas,
            weight_decay=1e-2,
            eps=eps,
        )
 
        # ── 11. LR scheduler ──────────────────────────────────────────────────
        self.use_lr_scheduler = eta_min_scheduler is not None
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                T_max=train_num_steps,
                eta_min=eta_min_scheduler,
            )
 
        # ── 12. Misc state ────────────────────────────────────────────────────
        self.results_folder = Path(results_folder)
        if self.is_main:
            self.results_folder.mkdir(exist_ok=True)
 
        self.step              = 0
        self.loss_history      = []
        self.test_loss_history = []
 
    # ── Properties ────────────────────────────────────────────────────────────
 
    @property
    def device(self):
        return self._device
 
    def _autocast(self):
        if self.amp:
            return torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype)
        return contextlib.nullcontext()
 
    # ── CP group injection ────────────────────────────────────────────────────
 
    def _inject_cp_group(self, diffusion_model, cp_group):
        """
        Walk the DiT blocks and set cp_group on every Attention module.
        Called once at init for both the training model and the EMA model.
        The attribute is read in Attention.forward() to decide whether to
        all-gather K/V. Setting it to None disables CP (single GPU mode).
        """
        dit = diffusion_model.neural_net  # GaussianDiffusion1D -> inner DiT
        for block in dit.blocks:
            attn = block.attn
            if isinstance(attn, Attention):
                attn.cp_group = cp_group if self.cp_size > 1 else None
 
    # ── Gradient hooks for DP all-reduce ──────────────────────────────────────
 
    def _register_dp_grad_hooks(self):
        """
        All-reduce gradients across DP replicas after the last accumulation
        step. Equivalent to DDP but without module wrapping.
 
        Why not DDP/replicate:
          - No DTensor complications (weights are plain tensors)
          - No module attribute conflicts with cp_group injection
          - Simpler and more transparent
        """
        dp_group = self.dp_mesh.get_group()
        dp_size  = self.dp_size
 
        def make_hook(p):
            def hook(grad):
                if self._dp_sync:
                    # div_ inside the hook: allowed during backward.
                    # Outside backward leaf grads are frozen, div_ would raise.
                    grad.div_(dp_size)
                    handle = dist.all_reduce(grad, group=dp_group, async_op=True)
                    self._dp_handles.append(handle)
                return grad
            return hook
 
        self._dp_hook_handles = []
        for p in self.model.parameters():
            if p.requires_grad:
                h = p.register_post_accumulate_grad_hook(make_hook(p))
                self._dp_hook_handles.append(h)
 
    def _wait_dp_grads(self):
        """Block until all async DP all-reduces have completed."""
        for handle in self._dp_handles:
            handle.wait()
        self._dp_handles.clear()
 
    # ── Sequence slicing ──────────────────────────────────────────────────────
 
    def _slice_for_cp(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Slice the full sequence for this CP rank.
 
        sequence: [B, total_seq_len, C] # esto creo que no es asi, mi input para el patcher es [B, C, seq_len]
        returns:  [B, total_seq_len // cp_size, C]
 
        IMPORTANT: total_seq_len must be divisible by cp_size.
        If your sequences are 260,000 tokens and cp_size=4, each rank gets
        65,000 tokens. Pad your dataset to the next multiple of cp_size if
        your sequence length is not evenly divisible.
        """
        total_len = sequence.shape[1]
        assert total_len % self.cp_size == 0, (
            f"Sequence length {total_len} must be divisible by cp_size "
            f"{self.cp_size}. Pad your dataset sequences accordingly."
        )
        chunk_size = total_len // self.cp_size
        start      = self.cp_rank * chunk_size
        # return sequence[:, start : start + chunk_size, :]
        return sequence[:, :, start : start + chunk_size]

 
    # ── Checkpoint ────────────────────────────────────────────────────────────
 
    def save(self, milestone):
        """
        Weights are replicated — only rank 0 writes. No collective needed.
        (Contrast with TP trainer where save() was collective due to DTensors.)
        """
        if not self.is_main:
            return
 
        torch.save(
            {
                'step':              self.step,
                'model':             self.model.state_dict(),
                'opt':               self.opt.state_dict(),
                'scheduler':         self.scheduler.state_dict() if self.use_lr_scheduler else None,
                'ema':               self.ema.state_dict(),
                'scaler':            self.scaler.state_dict() if self.scaler else None,
                'version':           __version__,
                'lr':                self.opt.param_groups[0]['lr'],
                'loss_history':      torch.tensor(self.loss_history),
                'test_loss_history': torch.tensor(self.test_loss_history),
            },
            str(self.results_folder / f'model-{milestone}.pt'),
        )
 
    def load(self, milestone):
        """
        All ranks load identical weights from the same file.
        Optimizer / scheduler state is restored on all ranks too so that
        gradient moments and LR are consistent everywhere.
        """
        data = torch.load(
            str(self.results_folder / f'model-{milestone}.pt'),
            map_location=self._device,
            weights_only=True,
        )
 
        # Strip possible "_orig_mod." prefix added by torch.compile
        state_dict = data['model']
        clean_sd   = {
            k.replace('_orig_mod.', '').replace('module.', ''): v
            for k, v in state_dict.items()
        }
        self.model.load_state_dict(clean_sd)
 
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
 
        if self.scaler and data.get('scaler'):
            self.scaler.load_state_dict(data['scaler'])
 
        if data.get('loss_history') is not None:
            self.loss_history = data['loss_history'].tolist()
        if data.get('test_loss_history') is not None:
            self.test_loss_history = data['test_loss_history'].tolist()
 
        if self.use_lr_scheduler and data.get('scheduler'):
            self.scheduler.load_state_dict(data['scheduler'])
 
        if 'lr' in data:
            for pg in self.opt.param_groups:
                pg['lr'] = data['lr']
            if self.is_main:
                print(f"Loaded lr: {data['lr']}")
 
        if 'version' in data and self.is_main:
            print(f"Loading from version {data['version']}")
 
    # ── Training loop ──────────────────────────────────────────────────────────
 
    def train(self):
        model    = self.model
        device   = self._device
        cp_group = self.cp_mesh.get_group()
 
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.is_main,
        ) as pbar:
 
            while self.step < self.train_num_steps:
                model.train()
                total_loss = 0.0
                self.opt.zero_grad()
 
                for acc_step in range(self.gradient_accumulate_every):
                    is_last_acc   = (acc_step == self.gradient_accumulate_every - 1)
                    # DP grad all-reduce fires only on the last accumulation step,
                    # identical to DDP's no_sync() pattern.
                    self._dp_sync = is_last_acc
 
                    sequence, classes = next(self.dl)
                    sequence = sequence.to(device)   # [B, total_seq_len, C]
                    classes  = classes.to(device)
 
                    # Each CP rank processes its own slice of the sequence.
                    # Attention.forward() all-gathers K/V across the cp_group
                    # so every rank attends over the full sequence context.
                    sequence_local = self._slice_for_cp(sequence)  # [B, seq/cp, C]
 
                    with self._autocast():
                        loss = model(sequence_local, classes=classes)
                        loss = loss / self.gradient_accumulate_every
 
                    # Average loss scalar across CP ranks for consistent logging.
                    # This does NOT affect gradients — only the reported number.
                    with torch.no_grad():
                        loss_scalar = loss.detach().clone()
                        dist.all_reduce(
                            loss_scalar, op=dist.ReduceOp.AVG, group=cp_group
                        )
 
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    # Gradient note:
                    # CP ranks do NOT all-reduce gradients across cp_group.
                    # Each CP rank computed a loss on seq/cp tokens, so its
                    # gradient covers that portion of the sequence. The weights
                    # are identical on all CP ranks, so after the optimizer step
                    # all CP ranks converge to the same update — this is exact,
                    # not an approximation, because the loss factorises over
                    # sequence positions (MSE / diffusion L2 losses).
                    # DP ranks DO all-reduce gradients (via hooks above) because
                    # they processed completely different samples.
 
                    total_loss += loss_scalar.item()
 
                # Wait for async DP all-reduces before touching gradients
                if self.scaler:
                    self.scaler.unscale_(self.opt)
                self._wait_dp_grads()
 
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )
 
                if self.scaler:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
 
                if self.use_lr_scheduler:
                    self.scheduler.step()
 
                self.step += 1
 
                # EMA runs on all ranks (weights replicated, EMA is identical
                # on every rank). cp_group already injected into ema_model attn.
                self.ema.update()
 
                if self.is_main:
                    self.loss_history.append(total_loss)
                    pbar.set_description(f'loss: {total_loss:.5f}')
                    pbar.update(1)
 
                # ── Periodic eval & checkpoint ─────────────────────────────────
                if self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
 
                    if self.dl_test is not None:
                        samples, sequences = self.eval_model(self.dl_test)
                        if self.is_main and samples is not None:
                            mse = ((samples - sequences) ** 2).mean().item()
                            self.test_loss_history.append(mse)
                            torch.save(
                                samples,
                                str(self.results_folder / f'sample-{milestone}.pt'),
                            )
 
                    # save() is rank-0 only — no barrier needed
                    self.save(milestone)
                    if self.is_main:
                        self.save_loss_plot()
 
        dist.barrier()
        if self.is_main:
            self.save_loss_plot()
            print('Training complete.')
        dist.destroy_process_group()
 
    # ── Evaluation ────────────────────────────────────────────────────────────
 
    @torch.no_grad()
    def eval_model(self, dl_test, **sampling_kwargs):
        """
        Run the EMA model over each DP replica's test shard.
        CP is transparent here: sample() calls Attention.forward() internally
        which will all-gather K/V just as during training.
 
        Predictions are gathered from all DP replicas onto rank 0.
        CP peers within a replica produce the same output (they share weights
        and process the same sample) so we only gather across dp_group.
        """
        self.ema.ema_model.eval()
        dp_group = self.dp_mesh.get_group()
        dp_size  = self.dp_size
 
        all_preds = []
        all_seqs  = []
 
        for data in tqdm(dl_test, disable=not self.is_main):
            sequence = data[0].to(self._device)
            classes  = data[1].to(self._device)
 
            with self._autocast():
                pred = self.ema.ema_model.sample(classes=classes, **sampling_kwargs)
 
            # Gather across DP replicas only
            gathered_pred = [torch.zeros_like(pred)     for _ in range(dp_size)]
            gathered_seq  = [torch.zeros_like(sequence) for _ in range(dp_size)]
            dist.all_gather(gathered_pred, pred,     group=dp_group)
            dist.all_gather(gathered_seq,  sequence, group=dp_group)
 
            if self.is_main:
                all_preds.append(torch.cat(gathered_pred, dim=0).cpu())
                all_seqs.append( torch.cat(gathered_seq,  dim=0).cpu())
 
        self.ema.ema_model.train()
 
        if self.is_main:
            return torch.cat(all_preds), torch.cat(all_seqs)
        return None, None
 
    # ── Plotting ───────────────────────────────────────────────────────────────
 
    def save_loss_plot(self):
        plt.figure()
        plt.plot(self.loss_history, label='Loss')
 
        if self.test_loss_history:
            test_x = list(range(
                self.save_and_sample_every,
                self.step + 1,
                self.save_and_sample_every,
            ))
            plt.plot(test_x, self.test_loss_history, label='Test Loss')
 
        window = 100
        if len(self.loss_history) >= window:
            ma = np.convolve(
                self.loss_history, np.ones(window) / window, mode='valid'
            )
            plt.plot(
                range(window - 1, len(self.loss_history)), ma,
                label=f'Moving Avg ({window})',
            )
 
        plt.yscale('log')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss Evolution')
        plt.legend()
        plt.savefig(
            self.results_folder / 'loss_evolution.png',
            bbox_inches='tight', pad_inches=0,
        )
        plt.close()