import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from denoising_diffusion_pytorch.autoencoder import AutoencoderKL1dConfig, AutoencoderKL1d, TrainerVAE1D
import shutil
import os
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)


@torch.no_grad()
def extract_latents(vae, dataset, batch_size=8, device="cuda",
                    num_workers=4, desc="Extracting latents"):
    """
    Encode a dataset to latents using the posterior mean (deterministic).
    Returns a float32 tensor of shape (N, latent_channels, L_down).
    """
    vae.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=True, num_workers=num_workers)
    latents = []
    for batch in tqdm(loader, desc=desc):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        z = vae.encode(x).mode()   # posterior mean — deterministic, no noise
        latents.append(z.cpu().float())

    return torch.cat(latents, dim=0)   # (N, C_lat, L_down)

@torch.inference_mode()
def save_latents(vae, dataset_train, dataset_test, save_dir,
                 batch_size=8, device="cuda", num_workers=4):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── extract raw latents ───────────────────────────────────────────────────
    z_train = extract_latents(vae, dataset_train, batch_size, device,
                              num_workers, desc="Train latents")
    z_test  = extract_latents(vae, dataset_test,  batch_size, device,
                              num_workers, desc="Test latents")

    print(f"Train latents: {tuple(z_train.shape)}")
    print(f"Test  latents: {tuple(z_test.shape)}")

    # ── statistics from train set only, per channel ───────────────────────────
    # mean/std over (N, L_down), keepdim so shape is (1, C_lat, 1) for broadcasting
    mean = z_train.mean(dim=(0, 2), keepdim=True)
    std  = z_train.std(dim=(0, 2),  keepdim=True).clamp(min=1e-6)

    print(f"\nPer-channel stats (train):")
    for c in range(mean.shape[1]):
        print(f"  ch {c}:  mean={mean[0,c,0]:.4f}  std={std[0,c,0]:.4f}")

    # ── standardise ───────────────────────────────────────────────────────────
    z_train_norm = (z_train - mean) / std
    z_test_norm  = (z_test  - mean) / std   # use TRAIN stats on test

    print(f"\nAfter standardisation:")
    print(f"  train — mean: {z_train_norm.mean():.6f}  std: {z_train_norm.std():.6f}")
    print(f"  test  — mean: {z_test_norm.mean():.6f}   std: {z_test_norm.std():.6f}")

    # ── save ──────────────────────────────────────────────────────────────────
    np.save(save_dir / "latents_train.npy", z_train_norm.numpy())
    np.save(save_dir / "latents_test.npy",  z_test_norm.numpy())
    np.savez(save_dir / "latents_stats.npz",
             mean=mean.numpy(),   # (1, C_lat, 1)
             std=std.numpy())     # (1, C_lat, 1)

    print(f"\nSaved to {save_dir}/")


def denormalise_latents(z: torch.Tensor, stats_path: str) -> torch.Tensor:
    """Apply before passing diffusion samples to vae.decode()."""
    stats = np.load(stats_path)
    mean  = torch.from_numpy(stats["mean"]).to(z)
    std   = torch.from_numpy(stats["std"]).to(z)
    return z * std + mean

@torch.no_grad()
def evaluate_vae(vae, dataset_test, cp_mean, cp_std, results_folder, batch_size=8, device="cuda", num_workers=4, original_length=260774):
    """
    Reconstruct the full test dataset and compute reconstruction metrics.

    Returns a dict with:
      - per_sample:  dict of arrays, one value per test sample
      - global:      dict of scalar metrics aggregated over the full dataset
      - originals:   (N, C, P) float32 numpy array
      - reconstructions: (N, C, P) float32 numpy array
    """
    vae = vae.to(device)
    vae.eval()

    loader = DataLoader(dataset_test, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)

    all_x     = []
    all_x_hat = []

    for batch in tqdm(loader, desc="Reconstructing"):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        # sample_posterior=False → deterministic, uses posterior mean (best reconstruction)
        x_hat, _ = vae(x, sample_posterior=False)
        x = x.cpu().float()[:, :, :original_length]
        x_hat = x_hat.cpu().float()[:, :, :original_length]
        x = x * cp_std + cp_mean
        x_hat = x_hat * cp_std + cp_mean
        all_x.append(x)
        all_x_hat.append(x_hat)

    all_x     = torch.cat(all_x,     dim=0)   # (N, C, P)
    all_x_hat = torch.cat(all_x_hat, dim=0)   # (N, C, P)
    torch.save(all_x_hat, results_folder + "/reconstructions.pt")
    # ── per-sample metrics ────────────────────────────────────────────────────
    diff   = all_x - all_x_hat                                   # (N, C, P)

    # MSE and MAE averaged over (C, P) for each sample
    mse_per_sample = diff.pow(2).mean(dim=(1, 2))                # (N,)
    mae_per_sample = diff.abs().mean(dim=(1, 2))                 # (N,)

    # R² per sample:  1 - SS_res / SS_tot
    ss_res = diff.pow(2).sum(dim=(1, 2))                         # (N,)
    ss_tot = (all_x - all_x.mean(dim=(1, 2), keepdim=True)
              ).pow(2).sum(dim=(1, 2))                           # (N,)
    r2_per_sample = 1.0 - ss_res / ss_tot.clamp(min=1e-8)       # (N,)

    # relative L2 error per sample
    rel_l2_per_sample = (
        diff.pow(2).sum(dim=(1, 2)).sqrt() /
        all_x.pow(2).sum(dim=(1, 2)).sqrt().clamp(min=1e-8)
    )                                                            # (N,)

    # ── per-channel metrics (averaged over N and P) ───────────────────────────
    mse_per_channel = diff.pow(2).mean(dim=(0, 2))               # (C,)
    mae_per_channel = diff.abs().mean(dim=(0, 2))                # (C,)

    ss_res_ch = diff.pow(2).sum(dim=(0, 2))
    ss_tot_ch = (all_x - all_x.mean(dim=(0, 2), keepdim=True)
                 ).pow(2).sum(dim=(0, 2))
    r2_per_channel = 1.0 - ss_res_ch / ss_tot_ch.clamp(min=1e-8)  # (C,)

    # ── global scalars ────────────────────────────────────────────────────────
    global_metrics = {
        "mse":        mse_per_sample.mean().item(),
        "rmse":       mse_per_sample.mean().sqrt().item(),
        "mae":        mae_per_sample.mean().item(),
        "r2":         r2_per_sample.mean().item(),
        "rel_l2":     rel_l2_per_sample.mean().item(),
        "max_error":  diff.abs().max().item(),
    }

    # ── pretty print ─────────────────────────────────────────────────────────
    print("\n── VAE reconstruction metrics ───────────────────────────────")
    for k, v in global_metrics.items():
        print(f"  {k:<12} {v:.6f}")

    C = all_x.shape[1]
    print("\n  Per-channel R²:")
    for c in range(C):
        print(f"    channel {c}:  R²={r2_per_channel[c]:.4f}  "
              f"RMSE={mse_per_channel[c].sqrt():.4f}  "
              f"MAE={mae_per_channel[c]:.4f}")
    print("─────────────────────────────────────────────────────────────\n")

    return {
        "per_sample": {
            "mse":    mse_per_sample.numpy(),
            "mae":    mae_per_sample.numpy(),
            "r2":     r2_per_sample.numpy(),
            "rel_l2": rel_l2_per_sample.numpy(),
        },
        "per_channel": {
            "mse": mse_per_channel.numpy(),
            "mae": mae_per_channel.numpy(),
            "r2":  r2_per_channel.numpy(),
        },
        "global":          global_metrics,
        "originals":       all_x.numpy(),
        "reconstructions": all_x_hat.numpy(),
    }

data = np.load("data/modulus_datasets_cylinder-flow_vv1/dataset/rawData.npy", allow_pickle=True)

fields = torch.from_numpy(data["x"]).float()
fields = fields.permute(0, 1, 3, 2) # put the channel dimension after the spatial dimensions (they where in last dim)
print("fields shape", fields.shape)

train_percent = 0.8
ntrain = int(train_percent*fields.shape[0])
ntest = fields.shape[0] - ntrain
fields_train = fields[:ntrain]
fields_test = fields[ntrain:]
fields_train = fields_train.reshape(-1, fields_train.shape[2], fields_train.shape[3]) # flatten the time dimensions
fields_test = fields_test.reshape(-1, fields_test.shape[2], fields_test.shape[3]) # flatten the time dimensions

# standardize the data per field independently
fields_train_mean = fields_train.mean(dim=(0, 2), keepdim=True)
fields_train_std = fields_train.std(dim=(0, 2), keepdim=True)
fields_train = (fields_train - fields_train_mean) / fields_train_std
fields_test = (fields_test - fields_train_mean) / fields_train_std

print("fields_train mean", fields_train.mean(dim=(0, 2)))
print("fields_train std", fields_train.std(dim=(0, 2)))

# add padding
original_length = fields_train.shape[-1]
target_length = 1704
pad_length = target_length - original_length
print("fields_train shape", fields_train.shape)
print("fields_test shape", fields_test.shape)

fields_train = F.pad(fields_train, (0, pad_length))
fields_test = F.pad(fields_test, (0, pad_length))

train_dataset = TensorDataset(fields_train)
test_dataset = TensorDataset(fields_test)


cfg = AutoencoderKL1dConfig(
    in_channels=3,
    base_channels=192,
    num_heads=8,
    qk_norm=True,
    attention_resolutions=[0, 1, 2, 3],
    channel_multipliers=[1, 2, 4, 4],
    latent_channels=16,
)
# cfg = AutoencoderKL1dConfig(
#     in_channels=4,
#     base_channels=192,
#     num_heads=8,
#     qk_norm=True,
#     attention_resolutions=[0, 1, 2, 3],
#     channel_multipliers=[1, 2, 4, 4],
#     latent_channels=8,
# )


model = AutoencoderKL1d(cfg)
print("number of parameters in the model: ", model.parameter_count())
results_folder = './results/vae_cylinder/first_vae_good_attn_good_split'

trainer = TrainerVAE1D(
    model,
    train_dataset,
    dataset_test=test_dataset,
    train_batch_size=16,
    gradient_accumulate_every=1,
    split_batches=True,
    train_num_steps=70000,
    results_folder=results_folder,
    save_every=5000,
    train_lr=1e-4,
    amp=True,
    mixed_precision_type='bf16',
    max_grad_norm=1.0,
    compile_model=True,
    eta_min_scheduler=1e-6
)

# trainer.load(results_folder + "/vae_best.pt")
trainer.train()
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
# trainer._save_loss_plot()


if trainer.accelerator.is_main_process:
    vae = trainer._unwrapped_model()
    results = evaluate_vae(vae, test_dataset, fields_train_mean, fields_train_std, batch_size=8,
                           device=trainer.device, results_folder=results_folder)

    # access anything you need
    r2_scores  = results["per_sample"]["r2"]       # (N,) — per-sample R²
    x_hat      = results["reconstructions"]        # (N, C, P) numpy array
    print(results["global"])  # dict of global scalar metrics
    print(f"Worst R² sample: {r2_scores.min():.4f} (idx {r2_scores.argmin()})")
    save_latents(vae, train_dataset, test_dataset,
                 save_dir=results_folder, batch_size=8, device=trainer.device)