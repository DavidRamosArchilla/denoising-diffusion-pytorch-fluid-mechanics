import os 
import json
from pathlib import Path
import shutil

import pyLOM.NN
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit import DiT_models, DiTCoordPE, DiT
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
from denoising_diffusion_pytorch.dit_graph import GraphDiT

import torch
from torch.utils.data import TensorDataset, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update({
#     "font.family": "serif",
#     "axes.edgecolor": "black",
#     "axes.linewidth": 1,
#     "legend.frameon": False,
#     "ytick.direction": "in",
#     "xtick.top": False,
#     "ytick.right": False,
#     "font.size": 12
# })

def get_split_indices(split_name, split_data, all_conds, subsplits=[],atol=1e-4, rtol=1e-12):
    if len(subsplits) > 0:
        data = []
        for subsplit_name in subsplits:
            data += split_data[split_name][subsplit_name]
    else:
        data = split_data[split_name]
    indices = []

    for row in data:
        # Compare this row against all rows in all_conds
        matches = np.all(np.isclose(all_conds, row, atol=atol, rtol=rtol), axis=1)
        found = np.where(matches)[0]

        if len(found) == 0:
            raise KeyError(f"No match found within tolerance for {row}")
        if len(found) > 1:
            raise ValueError(f"Multiple matches found within tolerance for {row}")

        indices.append(found[0])

    return indices

def sanity_check_splits(split_data, all_conds, subsplits=[]):
    train_idx = set(get_split_indices("train", split_data, all_conds=all_conds, subsplits=subsplits))
    val_idx   = set(get_split_indices("validation", split_data, all_conds=all_conds, subsplits=subsplits))
    test_idx  = set(get_split_indices("test", split_data, all_conds=all_conds))

    # Check pairwise disjointness
    assert train_idx.isdisjoint(val_idx),  "Train and Validation overlap!"
    assert train_idx.isdisjoint(test_idx), "Train and Test overlap!"
    assert val_idx.isdisjoint(test_idx),   "Validation and Test overlap!"

    # Optional: check coverage
    all_idx = train_idx | val_idx | test_idx
    assert len(all_idx) == len(split_data["all"]), \
        "Splits do not cover all samples exactly once"

    print("✅ Sanity check passed: splits are disjoint and complete.")

DATA_DIR  = Path("/home/airbus/CETACEO_cp_interp/DATA/TIFON")

# data = pyLOM.Dataset.load(DATA_DIR / "CADGroup_3_completo_stage_1.h5")
# data = pyLOM.Dataset.load("/home/airbus/CETACEO_cp_interp/CLUSTERING_TIFON/AIRBUS/database/CADGroup_3_completo_stage_1_reordered.h5")
data = pyLOM.Dataset.load("/home/airbus/CETACEO_cp_interp/DATA/TIFON/rans5/outputs/CADGroup_3_stage_1_reordered.h5")
mesh_coords = data.xyz[:, [0, 2]]
cp = data["BoundaryValues_CoefPressure"].T
cp = cp[:, np.newaxis, :]
alpha, mach = torch.tensor(data.get_variable('aoa')).float(), torch.tensor(data.get_variable('mach')).float()
print(f"Loaded data: cp shape {cp.shape}, alpha shape {alpha.shape}, mach shape {mach.shape}")

# split_data = np.load(DATA_DIR / 'best_train-val-test_split.npy', allow_pickle=True).item()
# with open(DATA_DIR / "tifon_split_complete.json") as f:
with open(DATA_DIR / "/home/airbus/CETACEO_cp_interp/TIFON/splitting/tifon_split_complete.json") as f:
    split_data = json.load(f)

all_conds = torch.stack((alpha, mach), dim=1).numpy()
train_indices = get_split_indices("train", split_data, all_conds, ["batch0", "batch1", "batch2"])
val_indices = get_split_indices("validation", split_data, all_conds, ["batch0", "batch1", "batch2"])
test_indices = get_split_indices("test", split_data, all_conds)
sanity_check_splits(split_data, all_conds, ["batch0", "batch1", "batch2"])

train_cp, train_conds = cp[train_indices], all_conds[train_indices]
val_cp, val_conds = cp[val_indices], all_conds[val_indices]
test_cp, test_conds = cp[test_indices], all_conds[test_indices]

cp_mean, cp_std = train_cp.mean(), train_cp.std()
train_cp = (train_cp - cp_mean) / cp_std
val_cp = (val_cp - cp_mean) / cp_std
test_cp = (test_cp - cp_mean) / cp_std

original_length = cp.shape[2]
pad_length = 13872 # divisible by 16
train_cp = np.pad(train_cp, ((0, 0), (0, 0), (0, pad_length - train_cp.shape[2])), mode='constant')
val_cp = np.pad(val_cp, ((0, 0), (0, 0), (0, pad_length - val_cp.shape[2])), mode='constant')
test_cp = np.pad(test_cp, ((0, 0), (0, 0), (0, pad_length - test_cp.shape[2])), mode='constant')
print(f"After padding, cp shapes: train {train_cp.shape}, val {val_cp.shape}, test {test_cp.shape}")

conds_mean, conds_std = train_conds.mean(axis=0), train_conds.std(axis=0)
print(f"Condition means: {conds_mean}, stds: {conds_std}")
train_conds = (train_conds - conds_mean) / conds_std
val_conds = (val_conds - conds_mean) / conds_std
test_conds = (test_conds - conds_mean) / conds_std
print(f"After normalization, condition means: {train_conds.mean(axis=0)}, stds: {train_conds.std(axis=0)}")

coefficients = {
    "cp_mean": float(cp_mean),
    "cp_std": float(cp_std),
    "conds_mean": conds_mean.tolist(),
    "conds_std": conds_std.tolist(),
    "original_length": int(original_length),
    "pad_length": int(pad_length),
}

train_dataset = TensorDataset(torch.tensor(train_cp).float(), torch.tensor(train_conds).float())
val_dataset = TensorDataset(torch.tensor(val_cp).float(), torch.tensor(val_conds).float())
test_dataset = TensorDataset(torch.tensor(test_cp).float(), torch.tensor(test_conds).float())

train_dataset_merged = ConcatDataset([train_dataset, val_dataset])

# model = Unet1D(
#     dim=128,
#     dim_mults=(1, 2, 4),  # , 8),
#     # flash_attn = False,
#     channels=1,  
#     cond_dim=2,
#     cond_drop_prob=0.2,
#     attn_dim_head=64,
#     attn_heads=8,
#     learn_sigma=False,
#     # self_condition=True,
#     # full_attn=True,
#     # qknorm=True
# )

model = DiT(
    depth=4,
    hidden_size=256,
    patch_size=8,
    num_heads=8,
    input_size=train_cp.shape[2], # dataset grid size
    cond_dim=2, # number of parameters (alpha, mach)
    class_dropout_prob=0.167,
    in_channels=1,
    learn_sigma=False,
    use_swiglu=True,
    # use_rope=True,
    # qk_norm=True, # when bf16 training
    attn_type="vanilla",  # window, linear, vanilla, physics
    # slice_num=64,
    mlp_ratio=4,
)

# model = GraphDiT(
#     mesh_pos=mesh_coords,
#     cond_dim=2,
#     class_dropout_prob=0.19,
#     in_channels=1,
#     learn_sigma=False,
#     hidden_size=512,
#     depth=7,
#     num_heads=8,
#     mlp_ratio=3.73,
#     learnable_slices=True,
#     num_patches=64,
#     static_prior=2.76,
#     temperature=0.2,
#     patch_knn=8,
#     reinject_pos=True,  
#     attn_type="vanilla",
# )

# model = DiTCoordPE(
#     coords=torch.tensor(coords, dtype=torch.float32),
#     depth=6,
#     hidden_size=128,
#     patch_size=1,
#     num_heads=4,
#     input_size=train_cp.shape[2], # dataset grid size
#     cond_dim=2, # number of parameters (alpha, mach)
#     class_dropout_prob=0.2,
#     in_channels=1,
#     learn_sigma=False,
#     use_swiglu=True,
#     # use_rope=True,
#     qk_norm=True, # when bf16 training
#     attn_type="vanilla",  # window, linear, vanilla
#     mlp_ratio=2.5,
# )

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print("Number trainable of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=train_cp.shape[2],
    cond_scale=2,
    num_sampling_steps=100,
    sampling_method="euler",
    # shifted_mu=1.0986
)

results_folder = 'results/tifon_final/batch2_vanilla_mlp_4'
os.makedirs(results_folder, exist_ok=True)
with open(results_folder + "/coefficients.json", "w") as f:
    json.dump(coefficients, f)

train_steps = 100000
trainer = Trainer1D(
    diffusion,
    dataset=train_dataset,
    dataset_test=val_dataset, # small_val_dataset is to avoid timeout when training on 2 GPUs
    train_batch_size=8,
    train_lr=0.0000893, # 0.0002, #1e-4,
    train_num_steps=train_steps,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    # amp=True,     # turn on mixed precision
    # mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    # use_cpu=True,
    use_muon=True,
    # compile_model=True,
    split_batches=True,
    muon_lr=1e-4,
    muon_adjust_lr_fn="match_rms_adamw",
    muon_weight_decay=1e-2,
)
# trainer.load(5)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
trainer.train()

samples, seqs = trainer.eval_model(test_dataset, batch_size=32, use_autocast=True)

if trainer.accelerator.is_main_process:

    test_data, test_parameters = test_dataset.tensors
    samples = samples[:, :original_length]  # Remove padding if it was added
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")

    from cetaceo.evaluators import RegressionEvaluator
    # preds = preds * (cp_max - cp_min) + cp_min
    samples = (samples * cp_std) + cp_mean
    test_data = (test_data * cp_std) + cp_mean

    evaluator = RegressionEvaluator()
    metrics = evaluator(samples, test_data[:, :original_length])
    # metrics = evaluator(preds, cp_test) # cp_test
    evaluator.print_metrics()