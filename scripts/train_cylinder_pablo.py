import h5py
import numpy as np
from pathlib import Path
import shutil
import os
from tqdm.auto import tqdm
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import Trainer1D
from denoising_diffusion_pytorch.video_dit import DiT_models, DiT
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/cylinder_pablo")
CHANNELS = ["Ux", "Uy", "p"]          # → axis 2, indices 0 / 1 / 2
N_FILES  = 159
DTYPE    = np.float32
# ─────────────────────────────────────────────────────────────────────────────

# Probe first file for (T, H, W)
with h5py.File(DATA_DIR / "0.h5", "r") as f:
    T, H, W = f[CHANNELS[0]].shape

print(f"Per-file shape : T={T}, H={H}, W={W}")
print(f"Tensor shape   : ({N_FILES}, {T}, {len(CHANNELS)}, {H}, {W})")
print(
    f"Memory estimate: "
    f"{N_FILES * T * len(CHANNELS) * H * W * np.dtype(DTYPE).itemsize / 1e9:.2f} GB"
)

# ── Load ──────────────────────────────────────────────────────────────────────
fields      = np.zeros((N_FILES, T, len(CHANNELS), H, W), dtype=DTYPE)
viscosities = np.zeros(N_FILES, dtype=DTYPE)
missing     = []

for i in tqdm(range(N_FILES), desc="Loading simulations"):
    fp = DATA_DIR / f"{i}.h5"
    if not fp.exists():
        missing.append(i)
        continue
    with h5py.File(fp, "r") as f:
        for c, ch in enumerate(CHANNELS):
            fields[i, :, c] = f[ch][:]
        viscosities[i] = f.attrs["nu"]

if missing:
    print(f"⚠  Missing files (left as zeros): {missing}")

fields      = torch.from_numpy(fields)       # (N, T, C, H, W)
viscosities = torch.from_numpy(viscosities).unsqueeze(-1)  # (N, 1)
print(f"\nLoaded — fields.shape = {fields.shape}, dtype = {fields.dtype}")
print(f"Viscosities shape: {viscosities.shape}, dtype = {viscosities.dtype}")
# ── Standardise: per channel, over (N, T, H, W) ──────────────────────────────
# nanmean / nanstd so that NaN cells don't corrupt the channel statistics.
# After normalisation any remaining NaNs are filled with 0 (= the channel mean).
REDUCE_DIMS = (0, 1, 3, 4)   # average over N, T, H, W — keep C

fields_mean = torch.nanmean(fields, dim=REDUCE_DIMS, keepdim=True)          # (1,1,C,1,1)
fields_std  = torch.sqrt(
    torch.nanmean((fields - fields_mean) ** 2, dim=REDUCE_DIMS, keepdim=True)
)

fields = (fields - fields_mean) / (fields_std + 1e-8)
fields = torch.nan_to_num(fields, nan=0.0)   # NaN → 0 (channel mean in std space)

viscosities_mean = viscosities.mean()
viscosities_std  = viscosities.std()
viscosities = (viscosities - viscosities_mean) / (viscosities_std + 1e-8)

print(f"NaNs after standardisation : {torch.isnan(fields).sum().item()}")
print(f"Channel means (should ≈ 0) : {fields.mean(dim=REDUCE_DIMS).squeeze().tolist()}")
print(f"Channel stds  (should ≈ 1) : {fields.std(dim=REDUCE_DIMS).squeeze().tolist()}")

# ── Train / test split  80 / 20, seed = 42 ───────────────────────────────────
N       = fields.shape[0]
n_train = int(0.8 * N)      # 127 for N=159
n_test  = N - n_train       #  32

rng  = torch.Generator().manual_seed(42)
perm = torch.randperm(N, generator=rng)

train_idx, test_idx = perm[:n_train], perm[n_train:]

fields_train      = fields[train_idx]       # (n_train, T, C, H, W)
fields_test       = fields[test_idx]        # (n_test,  T, C, H, W)
viscosities_train = viscosities[train_idx]  # (n_train,)
viscosities_test  = viscosities[test_idx]   # (n_test,)

print(f"\nSplit — train: {n_train} simulations  |  test: {n_test} simulations")

# ── Dataset ───────────────────────────────────────────────────────────────────
train_ds = TensorDataset(fields_train, viscosities_train)
test_ds  = TensorDataset(fields_test,  viscosities_test)

print(f"\ntrain_ds : {len(train_ds)} samples  — sample fields shape : {train_ds[0][0].shape}, viscosity shape: {train_ds[0][1].shape}")
print(f"test_ds  : {len(test_ds)} samples  — sample fields shape : {test_ds[0][0].shape}, viscosity shape: {test_ds[0][1].shape}")


model = DiT(
    depth=6,
    hidden_size=128,
    patch_size=2,
    num_frames=fields_train.shape[1],
    num_heads=4,
    input_size=tuple(fields_train.shape[-2:]), # dataset grid size
    cond_dim=1, # number of parameters (viscosity)
    class_dropout_prob=0.2,
    in_channels=fields_train.shape[2],
    learn_sigma=False,
    use_swiglu=True,
    # use_rope=True,
    qk_norm=True, # when bf16 training
    attn_type="vanilla",  # window, linear, vanilla, physics
    mlp_ratio=2.5,
    factorize=False,
)

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=tuple(fields_train.shape[-2:]),
    cond_scale=2,
    num_sampling_steps=100,
    sampling_method="euler",
    # shifted_mu=1.0986
)

results_folder = 'results/cylinder_pablo/first_xxs'

train_steps = 100000

trainer = Trainer1D(
    diffusion,
    dataset=train_ds,
    dataset_test=test_ds,
    train_batch_size=8,
    train_lr=1e-4,
    train_num_steps=train_steps,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,     # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    # use_cpu=True,
    use_muon=True,
    compile_model=True,
    split_batches=True
)

trainer.load(5)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
# trainer.train()

samples, seqs = trainer.eval_model(test_ds, batch_size=16, use_autocast=True) # , cfg_interval_start=0.2

if trainer.accelerator.is_main_process:

    test_data, test_parameters = test_ds.tensors
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")

    from cetaceo.evaluators import RegressionEvaluator
    # preds = preds * (cp_max - cp_min) + cp_min
    samples = (samples * fields_std) + fields_mean
    test_data = (test_data * fields_std) + fields_mean

    evaluator = RegressionEvaluator()
    metrics = evaluator(samples, test_data)
    # metrics = evaluator(preds, cp_test) # cp_test
    evaluator.print_metrics()