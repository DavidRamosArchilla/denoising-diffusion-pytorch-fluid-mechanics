import numpy as np
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit import DiT1D_models
import shutil
import os
import matplotlib.pyplot as plt


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

def get_split_indices(split_name, split_data):
    all_conditions = split_data["All"]
    data = split_data[split_name]
    indices = []
    for i in range(data.shape[0]):
        idx = np.where(all_conditions == data[i])[0][0]
        indices.append(idx)
    return indices

data = np.load("data/aeronef/db_random.npy", allow_pickle=True).item()
# cp = data["Cp"]
# train_size = 0.8
# training_indices = np.random.choice(cp.shape[0], int(cp.shape[0] * train_size), replace=False)
# test_indices = np.setdiff1d(np.arange(cp.shape[0]), training_indices)
split_data = np.load("data/aeronef/best_train-val-test_split.npy", allow_pickle=True).item()
training_indices = get_split_indices("Train", split_data)
val_indices = get_split_indices("Validation", split_data)
test_indices = get_split_indices("Test", split_data)

def load_dataset(indices, norm_coefficients, data):
    aoa = data["Alpha"][indices]
    vinf = data["Vinf"][indices]
    cp = data["Cp"][indices]
    if "cp_min" not in norm_coefficients:
        norm_coefficients["cp_min"] = cp.min()
        norm_coefficients["cp_max"] = cp.max()
    cp = (cp - norm_coefficients["cp_min"]) / (norm_coefficients["cp_max"] - norm_coefficients["cp_min"])

    if "vinf_mean" not in norm_coefficients:
        norm_coefficients["vinf_mean"] = vinf.mean()
        norm_coefficients["vinf_std"] = vinf.std()
    vinf = (vinf - norm_coefficients["vinf_mean"]) / norm_coefficients["vinf_std"]

    if "aoa_mean" not in norm_coefficients:
        norm_coefficients["aoa_mean"] = aoa.mean()
        norm_coefficients["aoa_std"] = aoa.std()
    aoa = (aoa - norm_coefficients["aoa_mean"]) / norm_coefficients["aoa_std"]

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
#     # learn_sigma=True,
#     # self_condition=True,
#     # full_attn = False
# )

model = DiT1D_models['DiT1D-S/1'](
    seq_len=data["Cp"].shape[1],
    cond_dim=2,
    class_dropout_prob=0.2,
    in_channels=1,
    # learn_sigma=True,
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=data["Cp"].shape[1],
    objective="pred_noise",  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=1000,
    timesteps=1000,  # number of steps
    # use_cfg_plus_plus=True,
    min_snr_loss_weight=True,
    min_snr_gamma=5
)


print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 1e9} GB")

results_folder = 'results/aeronef_cp_new_split/dit_S_1_rmsnorm'

train_steps = 150000

trainer = Trainer1D(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset,
    dataset_test=val_dataset,
    train_batch_size=32,
    train_lr=1e-4,
    num_samples=9,
    train_num_steps=train_steps+4,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.99,  # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=15000,
    max_grad_norm=1.0
    # use_cpu=True
)

# trainer.load(10)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
trainer.train()
# trainer.load(9)

# torch.cuda.empty_cache()  # Clear GPU memory
trainer.ema.ema_model.eval()  # Ensure eval mode
diffusion = trainer.accelerator.unwrap_model(diffusion)
diffusion.eval()

test_data, test_parameters = test_dataset.tensors
errors, samples = evaluate_model(
    trainer.ema.ema_model, # trainer.ema.ema_model, # diffusion
    test_parameters,
    test_data,
    32,
    cond_scale=6
)
print(f"Final errors:\n{errors}")
torch.save(samples, f"{results_folder}/test_predictions_ema.pt")