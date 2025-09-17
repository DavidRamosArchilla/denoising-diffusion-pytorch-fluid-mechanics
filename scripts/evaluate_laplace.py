import numpy as np
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import Unet, GaussianDiffusion, evaluate_model, Trainer
import shutil
import os


data = np.load("data/laplace/laplace_dataset_test_solutions.npy")
# at the moment the model expects inputs in [0, 1], like grayscale images 
data = (data - data.min()) / (data.max() - data.min())  # scale to [0, 1]
# data = data * 2 - 1  # scale
data = data[:, np.newaxis, :, :]  # add channel dimension   
parameters = np.load("data/laplace/laplace_dataset_test_parameters.npy")
# normalize parameters to [0, 1]
parameters = (parameters - parameters.min(axis=0)) / (parameters.max(axis=0) - parameters.min(axis=0))

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4),#, 8),
    # flash_attn = False,
    channels = 1,
    cond_dim=2,
    # full_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = (64, 64),
    objective = 'pred_noise',  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=50,
    timesteps = 1000,    # number of steps
)

trainer = Trainer(
    diffusion,
    # 'path/to/your/images',
    dataset=torch.arange(0, 200),
    train_batch_size = 32,
    train_lr = 8e-5,
    num_samples=9,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    results_folder = "results/laplace-conditioned",  # folder to save results to
    save_and_sample_every=2000,
    augment_horizontal_flip=False,
    # use_cpu=True
)

trainer.load(5)

errors = evaluate_model(
    diffusion,
    parameters,
    data,
    32,
)
print(f"Final errors:\n{errors}")