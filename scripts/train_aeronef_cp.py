import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit import DiT_models
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
import shutil
import os
import json
import matplotlib.pyplot as plt


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
# def get_split_indices(split_name, split_data):
#     all_conditions = split_data["All"]
#     data = split_data[split_name]
#     indices = []
#     for i in range(data.shape[0]):
#         idx = np.where(all_conditions == data[i])[0][0]
#         indices.append(idx)
#     return indices

def get_split_indices(split_name, split_data):
    all_conditions = split_data["All"]
    data = split_data[split_name]

    lookup = {tuple(row): i for i, row in enumerate(all_conditions)}
    return [lookup[tuple(row)] for row in data]

def sanity_check_splits(split_data):
    train_idx = set(get_split_indices("Train", split_data))
    val_idx   = set(get_split_indices("Validation", split_data))
    test_idx  = set(get_split_indices("Test", split_data))

    # Check pairwise disjointness
    assert train_idx.isdisjoint(val_idx),  "Train and Validation overlap!"
    assert train_idx.isdisjoint(test_idx), "Train and Test overlap!"
    assert val_idx.isdisjoint(test_idx),   "Validation and Test overlap!"

    # Optional: check coverage
    all_idx = train_idx | val_idx | test_idx
    assert len(all_idx) == len(split_data["All"]), \
        "Splits do not cover all samples exactly once"

    print("✅ Sanity check passed: splits are disjoint and complete.")

data = np.load("data/aeronef/db_random.npy", allow_pickle=True).item()
field_name_to_predict = "Cp" # "Pressure"
# cp = data["Cp"]
# train_size = 0.8
# training_indices = np.random.choice(cp.shape[0], int(cp.shape[0] * train_size), replace=False)
# test_indices = np.setdiff1d(np.arange(cp.shape[0]), training_indices)
split_data = np.load("data/aeronef/best_train-val-test_split.npy", allow_pickle=True).item()
training_indices = get_split_indices("Train", split_data)
val_indices = get_split_indices("Validation", split_data)
test_indices = get_split_indices("Test", split_data)
sanity_check_splits(split_data)
# import sys;sys.exit()

def load_dataset(indices, norm_coefficients, data):
    aoa = data["Alpha"][indices]
    vinf = data["Vinf"][indices]
    cp = data[field_name_to_predict][indices]
    # if "cp_min" not in norm_coefficients:
    #     norm_coefficients["cp_min"] = cp.min()
    #     norm_coefficients["cp_max"] = cp.max()
    # cp = (cp - norm_coefficients["cp_min"]) / (norm_coefficients["cp_max"] - norm_coefficients["cp_min"])
    if "cp_mean" not in norm_coefficients:
        norm_coefficients["cp_mean"] = cp.mean()
        norm_coefficients["cp_std"] = cp.std()
    cp = (cp - norm_coefficients["cp_mean"]) / norm_coefficients["cp_std"]

    if "vinf_mean" not in norm_coefficients:
        norm_coefficients["vinf_mean"] = vinf.mean()
        norm_coefficients["vinf_std"] = vinf.std()
    vinf = (vinf - norm_coefficients["vinf_mean"]) / norm_coefficients["vinf_std"]

    if "aoa_mean" not in norm_coefficients:
        norm_coefficients["aoa_mean"] = aoa.mean()
        norm_coefficients["aoa_std"] = aoa.std()
    aoa = (aoa - norm_coefficients["aoa_mean"]) / norm_coefficients["aoa_std"]

    # pad/truncate to length 27500
    target_len = 27500
    if field_name_to_predict == "Pressure" and cp.shape[1] < target_len:
        pad_width = target_len - cp.shape[1]
        cp = np.pad(cp, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    cp_t = torch.from_numpy(cp).float().unsqueeze(1)  # add channel dimension
    # cp_t = cp_t[..., :-1]
    aoa_t = torch.from_numpy(aoa).float()
    mach_t = torch.from_numpy(vinf).float()
    conditions = torch.stack([aoa_t, mach_t], dim=1)
    print(cp.min(), cp.max())
    print(aoa.mean(), aoa.std())
    print(vinf.mean(), vinf.std())
    print(cp_t.shape, conditions.shape)

    return TensorDataset(cp_t, conditions)

coefficients = {}
dataset = load_dataset(training_indices, coefficients, data)
val_dataset = load_dataset(val_indices, coefficients, data)
test_dataset = load_dataset(test_indices, coefficients, data)

# model = Unet1D(
#     dim=128,
#     dim_mults=(1, 2, 2, 4),  # , 8),
#     # flash_attn = False,
#     channels=1,  
#     cond_dim=2,
#     cond_drop_prob=0.5,
#     attn_dim_head=64,
#     attn_heads=8,
#     learn_sigma=False,
#     # self_condition=True,
#     # full_attn = False
# )
model = DiT_models['DiT-L/1'](
    input_size=dataset.tensors[0].shape[2],
    cond_dim=2,
    class_dropout_prob=0.5,
    in_channels=1,
    learn_sigma=False,
    # use_bias=False,
    use_swiglu=True,
    use_rope=False,
    qk_norm=True,
)

# diffusion = GaussianDiffusion1D(
#     model,
#     seq_length=dataset.tensors[0].shape[2],
#     objective="pred_noise",  # 'pred_noise' or 'pred_x0'
#     beta_schedule="cosine",
#     sampling_timesteps=1000,
#     timesteps=1000,  # number of steps
#     # use_cfg_plus_plus=True,
#     min_snr_loss_weight=True,
#     min_snr_gamma=5
# )

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=dataset.tensors[0].shape[2],
    cond_scale=6,
    num_sampling_steps=500,
    sampling_method="euler",
    # shifted_mu=1.0986
)
small_val_dataset = torch.utils.data.Subset(val_dataset, np.random.choice(len(val_dataset), 64, replace=False))

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 1e9} GB")
# TIENES LA NORMALIZACION DE LOS DATOS CAMBIADA
results_folder = 'results/aeronef_cp_new_split/FM_dit_L_bf16_qknorm'

train_steps = 200000
trainer = Trainer1D(
    diffusion,
    dataset=dataset,
    # dataset_test=val_dataset, # small_val_dataset is to avoid timout when training on 2 GPUs
    train_batch_size=64,
    train_lr=2e-4,
    num_samples=9,
    train_num_steps=train_steps+4,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,     # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=15000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    # use_cpu=True,
    # use_muon=True,
    # compile_model=True,
    split_batches=True
)

# trainer.load(21)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
with open(os.path.join(results_folder, 'norm_coefficients.json'), 'w') as f:
    json.dump(coefficients, f, indent=4)
trainer.train()

if trainer.accelerator.is_main_process:
    # torch.cuda.empty_cache()  # Clear GPU memory
    diffusion = trainer.accelerator.unwrap_model(diffusion, keep_torch_compile=True)
    trainer.ema.ema_model.eval()  # Ensure eval mode
    diffusion.eval()

    test_data, test_parameters = test_dataset.tensors
    errors, samples = evaluate_model(
        trainer.ema.ema_model, # trainer.ema.ema_model, # diffusion
        test_parameters,
        test_data,
        64,
        cond_scale=6
    )
    print(f"Final errors:\n{errors}")
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")