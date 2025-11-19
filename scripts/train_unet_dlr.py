import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D, Trainer1D

import shutil
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Get details about each GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 2**30:.1f} GB")

# Get current device
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Current device name: {torch.cuda.get_device_name()}")


data = np.load("data/dlr_airfoils/cp_train_small.npy")
# at the moment the model expects inputs in [0, 1], like grayscale images 
data_min, data_max = data.min(), data.max()
data = (data - data_min) / (data_max - data_min)  # scale to [0, 1]
# data = data.reshape(data.shape[0], 1, -1)  # concatenate the 2 channels into 1
print(data.shape)
# pad the sequences so they are divisible by 4 and i have no problems with the downsampling of the unet
# pad_width = ((0, 0),    # No padding for the N samples dimension
#              (0, 0),    # No padding for the 2 channels dimension
#              (1, 1))    # Add 1 element before and 1 after the sequence dimension
# data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)

parameters = np.load("data/dlr_airfoils/conditions_train_small.npy")
parameters_mean, parameters_std = parameters.mean(axis=0), parameters.std(axis=0)
parameters = (parameters - parameters_mean) / (parameters_std)

# load test data
test_data = np.load("data/dlr_airfoils/cp_test.npy")
test_data = (test_data - data_min) / (data_max - data_min) 
# test_data = test_data.reshape(test_data.shape[0], 1, -1)
# test_data = np.pad(test_data, pad_width=pad_width, mode='constant', constant_values=0)
test_parameters = np.load("data/dlr_airfoils/conditions_test.npy")
test_parameters = (test_parameters - parameters_mean) / (parameters_std)

dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(parameters, dtype=torch.float32))

model = Unet1D(
    dim = 64,
    dim_mults=(1, 2, 4), #, 8),
    # flash_attn = False,
    channels=2,
    cond_dim=2,
    # attn_heads=8,
    # full_attn=True,
    cond_drop_prob=0.0
)

results_dir = "results/dlr/unet"

training_iters = 50000
batch_size = 8
lr = 8e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def cycle(dl):
    while True:
        for data in dl:
            yield data


dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
dl = cycle(dl)
# optimizer
opt = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, fused=True)
# cosine annealing lr scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=training_iters, eta_min=1e-6)

steps = 0
all_losses = []
with tqdm(total=training_iters) as pbar:
    while steps < training_iters:
        data = next(dl)
        sequence, classes = data[0].to(device), data[1].to(device)
        preds = model(sequence, time=None, classes=classes)
        loss = F.mse_loss(preds, sequence)
        steps += 1
        all_losses.append(loss.item())
        pbar.set_description(f'loss: {loss.item():.4f}')
        pbar.update(1)