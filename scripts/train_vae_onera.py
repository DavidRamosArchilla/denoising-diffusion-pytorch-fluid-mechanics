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

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

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
    torch.save(all_x_hat, results_folder + "/resconstructions.pt")
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

qoi_list = ['cp', 'cfx', 'cfy', 'cfz'] # names of the quantites of interest
nwallp = 260774  # number of points on the aircraft skin

data_dir = "/home/airbus/onera_data"
Y_train_tot = np.load(data_dir + '/Ytrain.npy')
Y_test = np.load(data_dir + '/Ytest.npy')

df_description = pd.read_csv(data_dir + '/describe_train_test_repartition_with_weights.csv', index_col=0)
df_test = df_description.loc[~df_description['Train']]
df_train = df_description.loc[df_description['Train']]

ncase = len(df_description)  # 468
ntest = len(df_test) # 156
ntrain = ncase-ntest  # 312

# Create the output array to be of shape (ntrain, nwallp, 4)
Y_train_tot_conditions = np.array([Y_train_tot[nwallp*i:nwallp*(i+1),:] for i in range(ntrain)])
Y_test_tot_conditions = np.array([Y_test[nwallp*i:nwallp*(i+1),:] for i in range(ntest)])

print("Y_train_tot_conditions shape", Y_train_tot_conditions.shape)

Y_train = Y_train_tot_conditions
Y_test = Y_test_tot_conditions

# process with torch 
# first, move channels dim
Y_train = torch.tensor(Y_train, dtype=torch.float32).permute(0, 2, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).permute(0, 2, 1)

# normalize/standarize things
train_mean = Y_train.mean(dim=(0, 2), keepdim=True)
train_std  = Y_train.std(dim=(0, 2), keepdim=True)


Y_train = (Y_train - train_mean) / train_std
Y_test = (Y_test - train_mean) / train_std

# pad sequences to a multple of a power of 2. 260864 = 256 * 1019
original_length = Y_train.shape[2]
pad_length = 260776 # divisible by 8, the downsampling factor of the model
Y_train = F.pad(Y_train, (0, pad_length - nwallp))
Y_test = F.pad(Y_test, (0, pad_length - nwallp))

print("Y train shape", Y_train.shape)
print("Y test shape", Y_test.shape)
print("mean/std Y train test ", Y_train.mean(dim=(0, 2)), Y_test.mean(dim=(0, 2)), Y_train.std(dim=(0, 2)), Y_test.std(dim=(0, 2)))
dataset_train = TensorDataset(Y_train)

dataset_test = TensorDataset(Y_test)


cfg = AutoencoderKL1dConfig(
    in_channels=4,
    base_channels=192,
    num_heads=8,
    qk_norm=True,
    attention_resolutions=[0, 1, 2, 3]
)

model = AutoencoderKL1d(cfg)
print("number of parameters in the model: ", model.parameter_count())
results_folder = './results/vae_onera_full_attn'
trainer = TrainerVAE1D(
    model,
    dataset_train,
    dataset_test=dataset_test,
    train_batch_size=16,
    gradient_accumulate_every=2,
    split_batches=True,
    train_num_steps=70000,
    results_folder=results_folder,
    save_every=5000,
    train_lr=1e-4,
    # amlp=True,
    # mixed_precision_type='bf16',
    max_grad_norm=1.0,
    compile_model=True,
    eta_min_scheduler=1e-6
)

trainer.train()
# trainer.load("results/vae_onera/vae_best.pt")
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
# trainer._save_loss_plot()


if trainer.accelerator.is_main_process:
    vae = trainer._unwrapped_model()
    results = evaluate_vae(vae, dataset_test, train_mean, train_std, batch_size=8,
                           device=trainer.device, results_folder=results_folder)

    # access anything you need
    r2_scores  = results["per_sample"]["r2"]       # (N,) — per-sample R²
    x_hat      = results["reconstructions"]        # (N, C, P) numpy array
    print(results["global"])  # dict of global scalar metrics
    print(f"Worst R² sample: {r2_scores.min():.4f} (idx {r2_scores.argmin()})")