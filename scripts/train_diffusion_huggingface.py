import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, DDPMScheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import os

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

latents = np.load("data/non_linear_eq/non_linear_train_solutions.npy")   # (N, 4, H/8, W/8)
latents = latents[:, np.newaxis, :, :]
parameters = np.load("data/non_linear_eq/non_linear_train_parameters.npy")  # (N, 2)
parameters_mean, parameters_std = parameters.mean(axis=0), parameters.std(axis=0)
parameters = (parameters - parameters_mean) / (parameters_std)

class LatentDataset(Dataset):
    def __init__(self, latents, parameters):
        self.latents = torch.tensor(latents).float()
        self.parameters = torch.tensor(parameters).float()

    def __getitem__(self, idx):
        return {
            "latents": self.latents[idx],
            "parameters": self.parameters[idx],
        }

    def __len__(self):
        return self.latents.shape[0]

dataset = LatentDataset(latents, parameters)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="sigmoid")

latent_h, latent_w = latents.shape[-2:]
unet = UNet2DConditionModel(
    sample_size=latent_h,   # H/8
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"
    ),
    up_block_types=(
        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ),
    cross_attention_dim=32,
    resnet_time_scale_shift="scale_shift"
)

class ParameterProjector(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.model(x)

cond_projector = ParameterProjector(
    in_features=parameters.shape[1], hidden_features=128, out_features=32
)

params_to_optimize = list(unet.parameters()) + list(cond_projector.parameters())
optimizer = AdamW(params_to_optimize, lr=1e-4, fused=True)

accelerator = Accelerator(
    mixed_precision="no",
    gradient_accumulation_steps=4,
)

unet, cond_projector, optimizer, train_dataloader = accelerator.prepare(
    unet, cond_projector, optimizer, dataloader
)

num_epochs = 150
loss_history = []

for epoch in range(num_epochs):
    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_dataloader):
        clean_latents = batch['latents']
        params = batch['parameters']

        # 1. Sample noise and timesteps
        noise = torch.randn_like(clean_latents)
        bsz = clean_latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,),
            device=clean_latents.device
        ).long()
        noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

        # 2. Get conditioning embedding
        cond_embedding = cond_projector(params).unsqueeze(1)

        # 3. Predict the noise
        with accelerator.accumulate(unet):
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=cond_embedding
            ).sample

            # 4. Loss
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())
        loss_history.append(loss.item())

    progress_bar.close()

result_dir = "results/ldm_full_attention"
os.makedirs(result_dir, exist_ok=True)

if accelerator.is_main_process:
    unet.save_pretrained(f"{result_dir}/ldm_unet")
    cond_projector_path = f"{result_dir}/ldm_cond_projector.pt"
    torch.save(cond_projector.state_dict(), cond_projector_path)
    print(f"✅ Saved UNet to ldm_unet and projector to {cond_projector_path}")

    plt.figure(figsize=(8,4))
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Diffusion Training Loss (Log Scale)")
    plt.savefig(f"{result_dir}/diffusion_loss.png")
    plt.close()