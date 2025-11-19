import numpy as np
import torch
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

data = np.load("data/aeronef/db_random.npy", allow_pickle=True).item()
cp = data["Cp"]
train_size = 0.8
training_indices = np.random.choice(cp.shape[0], int(cp.shape[0] * train_size), replace=False)
test_indices = np.setdiff1d(np.arange(cp.shape[0]), training_indices)
cp_train = cp[training_indices]
cp_test = cp[test_indices]
cp_min, cp_max = cp_train.min(), cp_train.max()
cp_train = (cp_train - cp_min) / (cp_max - cp_min)
cp_test = (cp_test - cp_min) / (cp_max - cp_min)

alpha = data["Alpha"]
vinf = data["Vinf"]
alpha_train = alpha[training_indices]
vinf_train = vinf[training_indices]
alpha_test = alpha[test_indices]
vinf_test = vinf[test_indices]
# normalize parameters to mean 0, std 1
alpha_mean, alpha_std = alpha_train.mean(), alpha_train.std()
vinf_mean, vinf_std = vinf_train.mean(), vinf_train.std()
alpha_train = (alpha_train - alpha_mean) / alpha_std
vinf_train = (vinf_train - vinf_mean) / vinf_std    
alpha_test = (alpha_test - alpha_mean) / alpha_std
vinf_test = (vinf_test - vinf_mean) / vinf_std

dataset = TensorDataset(torch.tensor(cp_train, dtype=torch.float32).unsqueeze(1), # unsqueeze to add channel dimension
                        torch.tensor(np.stack([alpha_train, vinf_train], axis=1), dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(cp_test, dtype=torch.float32).unsqueeze(1), 
                             torch.tensor(np.stack([alpha_test, vinf_test], axis=1), dtype=torch.float32))

model = Unet1D(
    dim=64,
    dim_mults=(1, 2, 4),  # , 8),
    # flash_attn = False,
    channels=1,  
    cond_dim=2,
    cond_drop_prob=0.5,
    # attn_dim_head=64,
    # attn_heads=8,
    # learn_sigma=True,
    # self_condition=True,
    # full_attn = False
)

# model = DiT1D_models['DiT1D-S/1'](
#     seq_len=cp_train.shape[1],
#     cond_dim=2,
#     class_dropout_prob=0.2,
#     in_channels=1,
#     # learn_sigma=True,
# )

diffusion = GaussianDiffusion1D(
    model,
    seq_length=cp_train.shape[1],
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

results_folder = 'results/aeronef_cp/unet_bs64'

train_steps = 150000

trainer = Trainer1D(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset,
    dataset_test=test_dataset,
    train_batch_size=64,
    train_lr=1e-4,
    num_samples=9,
    train_num_steps=train_steps+4,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.99,  # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=10000,
    max_grad_norm=1.0
    # use_cpu=True
)

# trainer.load(10)
trainer.train()
# trainer.load(9)

# torch.cuda.empty_cache()  # Clear GPU memory
trainer.ema.ema_model.eval()  # Ensure eval mode
diffusion = trainer.accelerator.unwrap_model(diffusion)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
diffusion.eval()
