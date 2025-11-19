import numpy as np
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit import DiT1D_models, DiT1D
import shutil
import os
import matplotlib.pyplot as plt

import pyLOM


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

data_dir = "/home/d.ramos/Datos_DLR_pylom"

def load_dataset(name, norm_coefficients):
    data_train = pyLOM.Dataset.load(f"{data_dir}/{name}.h5")
    airfoil_coords = torch.tensor(data_train.xyz).float()
    aoa = data_train.get_variable('AoA')
    mach = data_train.get_variable('Mach')
    cp = data_train["CP"].T
    if "cp_min" not in norm_coefficients:
        norm_coefficients["cp_min"] = cp.min()
        norm_coefficients["cp_max"] = cp.max()
    cp = (cp - norm_coefficients["cp_min"]) / (norm_coefficients["cp_max"] - norm_coefficients["cp_min"])

    if "mach_mean" not in norm_coefficients:
        norm_coefficients["mach_mean"] = mach.mean()
        norm_coefficients["mach_std"] = mach.std()
    mach = (mach - norm_coefficients["mach_mean"]) / norm_coefficients["mach_std"]

    if "aoa_mean" not in norm_coefficients:
        norm_coefficients["aoa_mean"] = aoa.mean()
        norm_coefficients["aoa_std"] = aoa.std()
    aoa = (aoa - norm_coefficients["aoa_mean"]) / norm_coefficients["aoa_std"]

    cp_t = torch.from_numpy(cp).float().unsqueeze(1)  # add channel dimension
    # cp_t = cp_t[..., :-1]
    aoa_t = torch.from_numpy(aoa).float()
    mach_t = torch.from_numpy(mach).float()
    conditions = torch.stack([aoa_t, mach_t], dim=1)
    print(cp.min(), cp.max())
    print(aoa.mean(), aoa.std())
    print(mach.mean(), mach.std())
    print(cp_t.shape, conditions.shape)

    return TensorDataset(cp_t, conditions), airfoil_coords


norm_coefficients = {}
dataset_train, airfoil_coords = load_dataset("NRL7301_TRAIN", norm_coefficients)
dataset_test, _ = load_dataset("NRL7301_TEST", norm_coefficients)

data_ex = dataset_train[0]

model = DiT1D_models['DiT1D-S/1'](
    seq_len=data_ex[0].shape[1],
    cond_dim=2,
    class_dropout_prob=0.2,
    in_channels=1,
    learn_sigma=True,
)
# model = DiT1D(
#     depth=12,
#     hidden_size=768,
#     patch_size=1,
#     num_heads=12,
#     seq_len=data_ex[0].shape[1],
#     cond_dim=2,
#     class_dropout_prob=0.2,
#     in_channels=1,
#     learn_sigma=False,
# )

diffusion = GaussianDiffusion1D(
    model,
    seq_length=data_ex[0].shape[1], 
    objective='pred_noise',  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=1000,
    timesteps=1000,    # number of steps
    # use_cfg_plus_plus=False,
    min_snr_loss_weight=True,
    min_snr_gamma=5
)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 2**30} GB")

results_folder = 'results/dlr/dit_S_1_no_scheduler_grad_norm_learn_sigma'

trainer = Trainer1D(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset_train,
    dataset_test=dataset_test,
    train_batch_size=8,
    train_lr=1e-4,
    num_samples=9,
    train_num_steps=70001,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.99,  # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=10000,
    max_grad_norm=1,
    # use_cpu=True
)
torch.cuda.empty_cache()
trainer.load(4)
trainer.train()
# trainer.load(10)
trainer.ema.ema_model.eval() 
# diffusion = trainer.accelerator.unwrap_model(diffusion)
diffusion.eval()
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))

# errors, samples = evaluate_model(
#     trainer.ema.ema_model, # trainer.ema.ema_model, # diffusion
#     test_parameters,
#     test_data,
#     32,
#     cond_scale=6
# )
# print(f"Final errors:\n{errors}")
# torch.save(samples, f"{results_folder}/test_predictions_ema.pt")
