import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator, DataLoaderConfiguration
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from einops import rearrange
from pathlib import Path

from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt

class ResBlock1d(nn.Module):
    """
    ResNet block with GroupNorm + SiLU, identical pattern to diffusers ResnetBlock2D
    but with Conv1d.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.0):
        super().__init__()
        # self.norm1 = nn.RMSNorm(in_channels, eps=1e-6, elementwise_affine=True)
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        # self.norm2 = nn.RMSNorm(out_channels, eps=1e-6, elementwise_affine=True)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # activation before norm seems correct --> https://github.com/hkproj/pytorch-stable-diffusion/blob/6e6900078372af15eb1b13e73068cd563784f377/sd/decoder.py#L67
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = 32, qknorm=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = num_heads
        hidden_dim = dim_head * num_heads
        self.qk_norm = qknorm
        self.q_norm = nn.RMSNorm(hidden_dim) if qknorm else nn.Identity()
        self.k_norm = nn.RMSNorm(hidden_dim) if qknorm else nn.Identity()

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q = rearrange(q, 'b c n -> b n c') 
        k = rearrange(k, 'b c n -> b n c')
        v = rearrange(v, 'b c n -> b n c')
        dtype = q.dtype
        if self.qk_norm:
            q = self.q_norm(q.to(self.q_norm.weight.dtype)).to(dtype)
            k = self.k_norm(k.to(self.k_norm.weight.dtype)).to(dtype)
            
        q, k, v = map(lambda t: rearrange(t, 'b n (h c) -> b h c n', h = self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v) # this outputs (b, h, c, n)
        out = rearrange(out, 'b h c n -> b (h c) n')
        return self.to_out(out)


class Downsample1d(nn.Module):
    """Strided Conv1d (factor 2), same as diffusers Downsample2D."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """nearest-neighbour × 2 + Conv1d, same as diffusers Upsample2D."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
    

class DownBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res: int,
                 add_downsample: bool = True, add_attention: bool = False,
                 num_heads: int = 4, dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        resnets = []
        attns = []
        for i in range(num_res):
            resnets.append(ResBlock1d(in_ch if i == 0 else out_ch, out_ch,
                                     dropout=dropout))
            attns.append(
                Attention(out_ch, num_heads=num_heads, dim_head=out_ch // num_heads, qknorm=qk_norm)
                if add_attention else nn.Identity()
            )
        self.resnets = nn.ModuleList(resnets)
        self.attns = nn.ModuleList(attns)
        self.downsample = Downsample1d(out_ch) if add_downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res, attn in zip(self.resnets, self.attns):
            x = res(x)
            x = attn(x)
        return self.downsample(x)
    

class MidBlock1d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1,
                 dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        self.res1 = ResBlock1d(channels, channels, dropout=dropout)
        self.attn = Attention(channels, num_heads=num_heads, qknorm=qk_norm)
        self.res2 = ResBlock1d(channels, channels, dropout=dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x
    

class UpBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res: int,
                 add_upsample: bool = True, add_attention: bool = False,
                 num_heads: int = 4, dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        resnets = []
        attns = []
        for i in range(num_res):
            resnets.append(ResBlock1d(in_ch if i == 0 else out_ch, out_ch,
                                      dropout=dropout))
            attns.append(
                Attention(out_ch, num_heads=num_heads, dim_head=out_ch // num_heads, qknorm=qk_norm)
                if add_attention else nn.Identity()
            )
        self.resnets = nn.ModuleList(resnets)
        self.attns = nn.ModuleList(attns)
        self.upsample = Upsample1d(out_ch) if add_upsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res, attn in zip(self.resnets, self.attns):
            x = res(x)
            x = attn(x)
        return self.upsample(x)
    

class Encoder1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_multipliers: List[int],
        latent_channels: int,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = (),   # which block indices get attention
        num_heads: int = 1,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        channels = base_channels
        self.down_blocks = nn.ModuleList()
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            add_down = i < len(channel_multipliers) - 1     # no downsample on last
            self.down_blocks.append(DownBlock1d(
                in_ch=channels, out_ch=out_ch,
                num_res=num_res_blocks,
                add_downsample=add_down,
                add_attention=(i in attention_resolutions),
                num_heads=num_heads, dropout=dropout, qk_norm=qk_norm,
            ))
            channels = out_ch

        self.mid = MidBlock1d(channels, num_heads=num_heads,
                              dropout=dropout, qk_norm=qk_norm)
        # self.norm_out = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=True)
        self.norm_out = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.act = nn.SiLU()
        # outputs 2 × latent_channels so we can split into mean & log_var
        self.conv_out = nn.Conv1d(channels, latent_channels * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid(x)
        x = self.conv_out(self.act(self.norm_out(x)))
        return x     # (B, latent_channels*2, L_down)
    

class Decoder1d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        channel_multipliers: List[int],
        latent_channels: int,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = (),
        num_heads: int = 1,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        rev = list(reversed(channel_multipliers))
        inner_ch = base_channels * rev[0]

        self.conv_in = nn.Conv1d(latent_channels, inner_ch, 3, padding=1)
        self.mid = MidBlock1d(inner_ch, num_heads=num_heads,
                              dropout=dropout, qk_norm=qk_norm)

        self.up_blocks = nn.ModuleList()
        channels = inner_ch
        for i, mult in enumerate(rev):
            out_ch = base_channels * mult
            # mirror: last encoder level had no downsample → first decoder level has no upsample
            add_up = i > 0
            # attention mirrors encoder attention (reversed)
            orig_idx = len(channel_multipliers) - 1 - i
            self.up_blocks.append(UpBlock1d(
                in_ch=channels, out_ch=out_ch,
                num_res=num_res_blocks,
                add_upsample=add_up,
                add_attention=(orig_idx in attention_resolutions),
                num_heads=num_heads, dropout=dropout, qk_norm=qk_norm,
            ))
            channels = out_ch

        # self.norm_out = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=True)
        self.norm_out = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv1d(channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        x = self.mid(x)
        for blk in self.up_blocks:
            x = blk(x)
        x = self.conv_out(self.act(self.norm_out(x)))
        return x     # (B, out_channels, P)


class DiagonalGaussian:
    def __init__(self, params: torch.Tensor):
        """params: (B, 2*C, L) — first half is mean, second half is log_var."""
        self.mean, self.log_var = params.chunk(2, dim=1)
        self.log_var = self.log_var.clamp(-30.0, 20.0)
        self.std = (0.5 * self.log_var).exp()

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        """KL(q || N(0,I)) per element, summed over C and L, mean over B."""
        return 0.5 * (self.mean.pow(2) + self.log_var.exp() - 1.0 - self.log_var).sum()

    
@dataclass
class AutoencoderKL1dConfig:
    in_channels: int = 4
    """Number of input channels (C in your (N, C, P) data)."""

    base_channels: int = 128
    """Base feature width. Multiplied by channel_multipliers at each level."""

    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    """
    Per-level channel multipliers.  len() == number of encoder levels.
    With 4 levels the spatial dimension is downsampled by 2^3 = 8×
    (the last level has no downsample).
    Example: P=260000 → latent L=32500  (8× compression).
    Add more levels or increase multipliers for stronger compression.
    """

    latent_channels: int = 8
    """
    Channels in the latent z space.
    Smaller → more compressed, harder to reconstruct.
    8 or 16 is a good starting point.
    """

    num_res_blocks: int = 2
    """ResBlocks per encoder/decoder level."""

    attention_resolutions: List[int] = field(default_factory=lambda: [2, 3])
    """
    Encoder/decoder level indices that include self-attention.
    Deeper levels (smaller spatial size) are cheapest to attend over.
    """

    num_heads: int = 4
    """Attention heads."""

    qk_norm: bool = False

    dropout: float = 0.0

    kl_weight: float = 1e-6
    """
    Weight for KL term relative to reconstruction loss.
    Very small values (~1e-6 to 1e-4) make the VAE behave more like
    a regular AE with a regularised latent — good for LDM pre-training.
    Increase to get a better-structured latent space (at the cost of
    reconstruction quality).
    """

class AutoencoderKL1d(nn.Module):
    """
    1-D Variational AutoEncoder with KL regularisation.

    Mirrors the HuggingFace/diffusers AutoencoderKL API:

        model = AutoencoderKL1d(cfg)

        # encode
        posterior = model.encode(x)      # DiagonalGaussian
        z = posterior.sample()           # or .mode() for deterministic

        # decode
        x_hat = model.decode(z)

        # full forward (used during training)
        x_hat, posterior = model(x)
        loss, rec_loss, kl_loss = model.loss(x, x_hat, posterior)
    """

    def __init__(self, cfg: AutoencoderKL1dConfig):
        super().__init__()
        self.cfg = cfg

        # ── scaling constant (set after first forward pass, like diffusers) ──
        # multiply z by 1/scale_factor before passing to diffusion model so
        # that z has unit variance.  Call `model.set_scale_factor(loader)` once.
        self.register_buffer("scale_factor", torch.tensor(1.0))

        self.encoder = Encoder1d(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            channel_multipliers=cfg.channel_multipliers,
            latent_channels=cfg.latent_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=cfg.attention_resolutions,
            num_heads=cfg.num_heads,
            qk_norm=cfg.qk_norm,
            dropout=cfg.dropout,
        )
        self.decoder = Decoder1d(
            out_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            channel_multipliers=cfg.channel_multipliers,
            latent_channels=cfg.latent_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=cfg.attention_resolutions,
            num_heads=cfg.num_heads,
            qk_norm=cfg.qk_norm,
            dropout=cfg.dropout,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> DiagonalGaussian:
        """x: (B, C, P) → DiagonalGaussian over (B, latent_channels, P_down)."""
        params = self.encoder(x)
        return DiagonalGaussian(params)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_channels, P_down) → (B, C, P)."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussian]:
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        x_hat = self.decode(z)
        return x_hat, posterior

    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        posterior: DiagonalGaussian,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, reconstruction_loss, kl_loss).
        rec_loss: mean L2 over all elements.
        kl_loss:  KL divergence, normalised per element.
        """
        rec_loss = F.mse_loss(x_hat, x)
        kl_loss = posterior.kl() / x.numel()          # normalise per element
        total = rec_loss + self.cfg.kl_weight * kl_loss
        return total, rec_loss, kl_loss

    # ── helpers ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def set_scale_factor(self, loader, device="cuda", num_batches: int = 16):
        """
        Compute the std of encoder means on a few batches and store as
        scale_factor (like diffusers does for SD).
        Call once after training, before training the diffusion model.
        """
        self.eval()
        stds = []
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            x = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
            z = self.encode(x).mode()
            stds.append(z.std().item())
        self.scale_factor = torch.tensor(sum(stds) / len(stds), device=self.scale_factor.device)
        print(f"[AutoencoderKL1d] scale_factor set to {self.scale_factor.item():.4f}")

    def latent_shape(self, input_length: int) -> Tuple[int, int]:
        """Returns (latent_channels, latent_length) for a given input length."""
        num_down = len(self.cfg.channel_multipliers) - 1
        return self.cfg.latent_channels, input_length // (2 ** num_down)

    def parameter_count(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"{total / 1e6:.2f}M"

def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

class TrainerVAE1D:
    def __init__(
        self,
        model: AutoencoderKL1d,
        dataset,
        *,
        # ── data ──────────────────────────────────────────────────
        train_batch_size: int = 16,
        dataset_test=None,
        # ── optimisation ──────────────────────────────────────────
        train_lr: float = 1e-4,
        adam_betas: tuple = (0.9, 0.999),
        adam_weight_decay: float = 1e-2,
        train_num_steps: int = 100_000,
        gradient_accumulate_every: int = 1,
        max_grad_norm: float | None = 1.0,
        # ── loss ──────────────────────────────────────────────────
        kl_weight: float | None = None,
        # ── lr scheduler ──────────────────────────────────────────
        eta_min_scheduler: float | None = None,
        # ── mixed precision / accelerate ──────────────────────────
        amp: bool = False,
        mixed_precision_type: str = "bf16",
        split_batches: bool = True,
        use_cpu: bool = False,
        # ── checkpointing ─────────────────────────────────────────
        results_folder: str = "./results_vae",
        save_every: int = 1000,
        # ── misc ──────────────────────────────────────────────────
        compile_model: bool = False,
        num_workers: int = 4,
    ):
        super().__init__()

        # ── accelerator ───────────────────────────────────────────────────────
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision_type if amp else "no",
            cpu=use_cpu,
            dataloader_config=DataLoaderConfiguration(split_batches=split_batches),
            gradient_accumulation_steps=gradient_accumulate_every,
        )

        # ── model ─────────────────────────────────────────────────────────────
        self.model = model

        # kl_weight can be overridden here without changing the config
        if exists(kl_weight):
            self.model.cfg.kl_weight = kl_weight

        # ── training state ────────────────────────────────────────────────────
        self.step = 0
        self.train_num_steps = train_num_steps
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every

        # ── data ──────────────────────────────────────────────────────────────
        dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        if exists(dataset_test):
            self.dl_test = DataLoader(
                dataset_test,
                batch_size=train_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            self.dl_test = None

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # ── optimiser ─────────────────────────────────────────────────────────
        # eps follows mixed-precision convention (same as the original Trainer1D)
        eps = 1e-6 if amp and mixed_precision_type in ("fp16", "bf16") else 1e-8
        self.opt = AdamW(
            model.parameters(),
            lr=train_lr,
            betas=adam_betas,
            weight_decay=adam_weight_decay,
            fused=not use_cpu,   # fused kernel only available on CUDA
            eps=eps,
        )

        # ── lr scheduler ──────────────────────────────────────────────────────
        self.use_lr_scheduler = exists(eta_min_scheduler)
        if self.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                T_max=train_num_steps,
                eta_min=eta_min_scheduler,
            )
            self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)

        # ── results folder ────────────────────────────────────────────────────
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        # ── prepare everything with accelerate ────────────────────────────────
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if exists(self.dl_test):
            self.dl_test = self.accelerator.prepare(self.dl_test)

        if compile_model:
            self.accelerator.print("Compiling model …")
            self.model = torch.compile(self.model)
            self.accelerator.print("Model compiled.")

        # ── loss history ──────────────────────────────────────────────────────
        self.loss_history: list[dict] = []      # {"step", "rec", "kl", "total"}
        self.test_loss_history: list[dict] = [] # {"step", "rec", "kl", "total"}

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def _unwrapped_model(self) -> AutoencoderKL1d:
        return self.accelerator.unwrap_model(self.model)

    # ── checkpointing ─────────────────────────────────────────────────────────

    def save(self, step: int | None = None, tag: str | None = None):
        """
        Save model weights + optimiser + scheduler + loss history.

        Filenames:
            vae_{step:08d}.pt   if step is given
            vae_{tag}.pt        if a custom tag is given (e.g. 'best')
            vae_latest.pt       always updated so you can resume easily
        """
        if not self.is_main:
            return
        lr = self.opt.param_groups[0]['lr']
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "loss_history": self.loss_history,
            "test_loss_history": self.test_loss_history,
            "lr": lr,
        }
        if self.use_lr_scheduler:
            data["scheduler"] = self.scheduler.state_dict()

        # always write a "latest" file for easy resuming
        torch.save(data, self.results_folder / "vae_latest.pt")

        if exists(step):
            torch.save(data, self.results_folder / f"vae_{step:08d}.pt")
        if exists(tag):
            torch.save(data, self.results_folder / f"vae_{tag}.pt")

    def load(self, path: str | Path | None = None):
        """
        Load a checkpoint.  If *path* is None, looks for vae_latest.pt in
        results_folder.
        """
        path = Path(path) if exists(path) else self.results_folder / "vae_latest.pt"

        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")

        data = torch.load(path, map_location=self.device)

        unwrapped = self._unwrapped_model()
        unwrapped.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])

        if self.use_lr_scheduler and "scheduler" in data:
            self.scheduler.load_state_dict(data["scheduler"])

        self.loss_history      = data.get("loss_history", [])
        self.test_loss_history = data.get("test_loss_history", [])

        if "lr" in data:
            print(f"Setting loaded learning rate to {data['lr']}")
            for param_group in self.opt.param_groups:
                param_group['lr'] = data['lr']

        self.accelerator.print(f"Loaded checkpoint from {path}  (step {self.step})")

    # ── plotting ──────────────────────────────────────────────────────────────

    def _save_loss_plot(self):
        """Save a PNG with train (and optional test) rec / kl / total losses."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"VAE losses — step {self.step}", fontsize=12)

        metrics = ("rec", "kl", "total")
        titles  = ("Reconstruction loss", "KL loss", "Total loss")

        for ax, key, title in zip(axes, metrics, titles):
            if self.loss_history:
                steps_tr = [e["step"] for e in self.loss_history]
                vals_tr  = [e[key]   for e in self.loss_history]
                ax.plot(steps_tr, vals_tr, linewidth=0.8, alpha=0.6, label="train")

                # smoothed train curve (simple moving average)
                if len(vals_tr) >= 50:
                    window = max(1, len(vals_tr) // 50)
                    smoothed = [
                        sum(vals_tr[max(0, i - window): i + 1]) /
                        len(vals_tr[max(0, i - window): i + 1])
                        for i in range(len(vals_tr))
                    ]
                    ax.plot(steps_tr, smoothed, linewidth=1.5, label="train (smooth)")

            if self.test_loss_history:
                steps_te = [e["step"] for e in self.test_loss_history]
                vals_te  = [e[key]   for e in self.test_loss_history]
                ax.plot(steps_te, vals_te, "o-", linewidth=1.5,
                        markersize=4, label="test")

            ax.set_title(title)
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            # use log scale for kl which can vary by orders of magnitude
            if key == "kl" and self.loss_history:
                vals = [e[key] for e in self.loss_history if e[key] > 0]
                # if vals:
                #     ax.set_yscale("log")

        plt.tight_layout()
        # plot_path = self.results_folder / f"losses_step{self.step:08d}.png"
        latest_path = self.results_folder / "losses_latest.png"
        # plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.savefig(latest_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── validation helper ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _eval_test(self) -> dict | None:
        if not exists(self.dl_test):
            return None

        self.model.eval()
        total_rec = total_kl = total_loss = 0.0
        n = 0

        for batch in self.dl_test:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_hat, posterior = self.model(x)
            loss, rec, kl = self._unwrapped_model().loss(x, x_hat, posterior)

            # gather across processes
            rec_g, kl_g, loss_g = self.accelerator.gather_for_metrics(
                (rec.unsqueeze(0), kl.unsqueeze(0), loss.unsqueeze(0))
            )
            total_rec  += rec_g.mean().item()
            total_kl   += kl_g.mean().item()
            total_loss += loss_g.mean().item()
            n += 1

        self.model.train()
        return {"rec": total_rec / n, "kl": total_kl / n, "total": total_loss / n}

    # ── main training loop ────────────────────────────────────────────────────

    def train(self):
        accelerator = self.accelerator
        model = self.model
        cfg = self._unwrapped_model().cfg

        accelerator.print(
            f"\nStarting training: steps={self.train_num_steps}  "
            f"batch={self.batch_size}  "
            f"grad_accum={self.gradient_accumulate_every}  "
            f"kl_weight={cfg.kl_weight:.2e}\n"
        )

        best_test_loss = float("inf")

        # tqdm only on the main process so it doesn't duplicate across GPUs
        pbar = tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.is_main,
            dynamic_ncols=True,
            desc="Training VAE",
        )

        model.train()
        while self.step < self.train_num_steps:
            # ── gradient accumulation loop ────────────────────────────────────
            acc_rec = acc_kl = acc_total = 0.0

            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dl)
                x = batch[0] if isinstance(batch, (list, tuple)) else batch

                with accelerator.accumulate(model):
                    x_hat, posterior = model(x)
                    loss, rec, kl = self._unwrapped_model().loss(x, x_hat, posterior)
                    accelerator.backward(loss)

                acc_rec   += rec.item()
                acc_kl    += kl.item()
                acc_total += loss.item()

            # average over accumulation steps
            acc_rec   /= self.gradient_accumulate_every
            acc_kl    /= self.gradient_accumulate_every
            acc_total /= self.gradient_accumulate_every

            # ── gradient clipping & optimiser step ────────────────────────────
            if exists(self.max_grad_norm):
                accelerator.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

            if self.use_lr_scheduler:
                self.scheduler.step()

            self.step += 1

            # ── logging / progress bar ────────────────────────────────────────
            if self.is_main:
                lr_now = self.opt.param_groups[0]["lr"]
                self.loss_history.append(
                    {"step": self.step, "rec": acc_rec, "kl": acc_kl, "total": acc_total}
                )
                pbar.set_postfix(
                    rec=f"{acc_rec:.4f}",
                    kl=f"{acc_kl:.4f}",
                    total=f"{acc_total:.4f}",
                    lr=f"{lr_now:.2e}",
                )
            pbar.update(1)

            # ── periodic checkpoint + optional eval ───────────────────────────
            if self.step % self.save_every == 0:
                # eval on test set
                test_metrics = self._eval_test()

                if self.is_main:
                    if exists(test_metrics):
                        self.test_loss_history.append(
                            {"step": self.step, **test_metrics}
                        )
                        pbar.write(
                            f"[step {self.step}] test →  "
                            f"rec {test_metrics['rec']:.4f}  "
                            f"kl {test_metrics['kl']:.4f}  "
                            f"total {test_metrics['total']:.4f}"
                        )
                        if test_metrics["total"] < best_test_loss:
                            best_test_loss = test_metrics["total"]
                            self.save(tag="best")
                            pbar.write(
                                f"  ↳ new best test loss {best_test_loss:.5f} — saved vae_best.pt"
                            )

                    # numbered + latest checkpoint
                    self.save(step=self.step)
                    pbar.write(f"  ↳ checkpoint saved (step {self.step})")

                    # loss plot
                    self._save_loss_plot()
                    pbar.write(
                        f"  ↳ loss plot saved → {self.results_folder / 'losses_latest.png'}"
                    )

        # ── end of training ───────────────────────────────────────────────────
        pbar.close()
        if self.is_main:
            self.save(tag="final")
            self._save_loss_plot()
            accelerator.print("Training complete. Saved vae_final.pt + final loss plot.")
        accelerator.wait_for_everyone()
