import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from tqdm.auto import tqdm


data = np.load("data/non_linear_eq/train_high_resolution_solutions.npy")
data_min, data_max = data.min(), data.max()
data = (data - data_min) / (data_max - data_min)
data = data * 2.0 - 1.0
data = data[:, np.newaxis, :, :]

parameters = np.load("data/non_linear_eq/train_high_resolution_parameters.npy")
parameters_mean, parameters_std = parameters.mean(axis=0), parameters.std(axis=0)
parameters = (parameters - parameters_mean) / parameters_std


class CustomDataset(Dataset):
    def __init__(self, solutions, parameters):
        self.solutions = torch.tensor(solutions).float()
        self.parameters = torch.tensor(parameters).float()

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, idx):
        return self.solutions[idx], self.parameters[idx]

dataset = CustomDataset(data, parameters)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)


vae = AutoencoderKL.from_pretrained("vae_ldm/final")
vae.eval().cuda()

scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)


all_latents = []
all_params = []

with torch.no_grad():
    for imgs, params in tqdm(dataloader, desc="Encoding latents"):
        imgs = imgs.cuda().to(dtype=torch.float32)
        latents = vae.encode(imgs).latent_dist.sample()
        latents = latents * scaling_factor  # scale for diffusion
        all_latents.append(latents.cpu().numpy())
        all_params.append(params.numpy())

all_latents = np.concatenate(all_latents, axis=0)   # shape (N, 4, H/8, W/8)
all_params = np.concatenate(all_params, axis=0)     # shape (N, param_dim)

os.makedirs("data/non_linear_eq_latents", exist_ok=True)
np.save("data/non_linear_eq_latents/train_latents.npy", all_latents)
np.save("data/non_linear_eq_latents/train_parameters.npy", all_params)

print("Saved precomputed latents and parameters")
print("Latents shape:", all_latents.shape)
