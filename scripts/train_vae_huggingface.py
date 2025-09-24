import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

data = np.load("data/non_linear_eq/train_high_resolution_solutions.npy")
data_min, data_max = data.min(), data.max()
data = (data - data_min) / (data_max - data_min)   # scale to [0,1]
data = data * 2.0 - 1.0                           # rescale to [-1,1]
data = data[:, np.newaxis, :, :]                  # add channel dim

parameters = np.load("data/non_linear_eq/train_high_resolution_parameters.npy")
parameters_mean, parameters_std = parameters.mean(axis=0), parameters.std(axis=0)
parameters = (parameters - parameters_mean) / parameters_std


class CustomDataset(Dataset):
    def __init__(self, solutions, parameters):
        super().__init__()
        self.solutions = torch.tensor(solutions).float()
        self.parameters = torch.tensor(parameters).float()

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, idx):
        return {
            "pixel_values": self.solutions[idx],
            "parameters": self.parameters[idx],
        }


dataset = CustomDataset(data, parameters)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)  # small per-GPU batch

image_size = data.shape[-1]

vae = AutoencoderKL(
    in_channels=1,
    out_channels=1,
    latent_channels=4,
    sample_size=image_size,
    block_out_channels=(64, 128, 128),
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    norm_num_groups=8,
)

vae.config.scaling_factor = 0.18215
vae.config.downsample_factor = 8

grad_accum_steps = 4   # effective batch size = 4 * 4 = 16
accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)
device = accelerator.device

optimizer = AdamW(vae.parameters(), lr=1e-4, weight_decay=0.0)
num_epochs = 40
kl_weight = 1e-6
log_every = 100
output_dir = "vae_ldm"
os.makedirs(output_dir, exist_ok=True)

vae, optimizer, dataloader = accelerator.prepare(vae, optimizer, dataloader)
vae.train()

global_step = 0
loss_history = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for step, batch in enumerate(pbar):
        with accelerator.accumulate(vae):  
            pixel_values = batch["pixel_values"].to(device)

            # encode/decode
            encode_out = vae.encode(pixel_values)
            latent_dist = encode_out.latent_dist
            z = latent_dist.sample()
            recon = vae.decode(z).sample

            # losses
            recon_loss = F.mse_loss(recon, pixel_values)
            mu, logvar = latent_dist.mean, latent_dist.logvar
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]
            ).mean()

            loss = recon_loss + kl_weight * kl_loss

            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.detach().item()
        loss_history.append(loss.item())
        global_step += 1

        if global_step % log_every == 0:
            accelerator.print(
                f"step {global_step} | recon: {recon_loss.item():.6f} | kl: {kl_loss.item():.6f}"
            )

    avg_epoch_loss = epoch_loss / len(dataloader)
    accelerator.print(f"Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.6f}")

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(vae)
        unwrapped.save_pretrained(os.path.join(output_dir, f"epoch_{epoch+1}"))

# save final
if accelerator.is_main_process:
    unwrapped = accelerator.unwrap_model(vae)
    unwrapped.save_pretrained(os.path.join(output_dir, "final"))
accelerator.wait_for_everyone()

if accelerator.is_main_process:
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, label="train loss", alpha=0.7)
    # add smoothed curve for readability
    if len(loss_history) > 50:
        window = 50
        smoothed = np.convolve(loss_history, np.ones(window)/window, mode="valid")
        plt.plot(range(window-1, window-1+len(smoothed)), smoothed, label="smoothed", linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("VAE training loss evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

vae.eval()
batch = next(iter(dataloader))
pixel_values = batch["pixel_values"].to(device)
with torch.no_grad():
    encode_out = vae.encode(pixel_values)
    z = encode_out.latent_dist.mean
    recon = vae.decode(z).sample.detach().cpu()
orig = pixel_values.detach().cpu()

n = min(4, recon.shape[0])
fig, axes = plt.subplots(n, 2, figsize=(6, 3*n))
for i in range(n):
    axes[i,0].imshow(orig[i,0], cmap="gray", vmin=-1, vmax=1)
    axes[i,0].set_title("original")
    axes[i,0].axis("off")
    axes[i,1].imshow(recon[i,0], cmap="gray", vmin=-1, vmax=1)
    axes[i,1].set_title("reconstruction")
    axes[i,1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "recon_examples.png"))
plt.close()
