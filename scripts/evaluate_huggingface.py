import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# from train_diffusion_huggingface import ParameterProjector


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

data = np.load("data/non_linear_eq/non_linear_train_solutions.npy")
data_min, data_max = data.min(), data.max()
data = (data - data_min) / (data_max - data_min) 
parameters = np.load("data/non_linear_eq_latents/train_parameters.npy") 
# normalize parameters to [0, 1]
parameters_mean, parameters_std = parameters.mean(axis=0), parameters.std(axis=0)
parameters = (parameters - parameters_mean) / (parameters_std)

test_data = np.load("data/non_linear_eq/non_linear_test_solutions.npy")
test_data = (test_data - data_min) / (data_max - data_min) 
test_data = test_data[:, np.newaxis, :, :]
test_parameters = np.load("data/non_linear_eq/non_linear_test_parameters.npy")
test_parameters = (test_parameters - parameters_mean) / (parameters_std)
test_parameters = torch.tensor(test_parameters).float()

results_dir = "results/ldm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet2DConditionModel.from_pretrained(f"{results_dir}/ldm_unet").to(device)
unet.eval()

cond_projector = ParameterProjector(
    in_features=test_parameters.shape[1], hidden_features=128, out_features=32
).to(device)
cond_projector.load_state_dict(torch.load(f"{results_dir}/ldm_cond_projector.pt", map_location=device))
cond_projector.eval()

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

@torch.no_grad()
def sample_from_params(params, num_inference_steps=1000):
    bsz = params.shape[0]

    # 1. Start from pure Gaussian noise in latent space
    h = unet.config.sample_size
    w = unet.config.sample_size
    latents = torch.randn((bsz, 1, h, w), device=device)

    # 2. Conditioning embedding
    cond_embedding = cond_projector(params.to(device)).unsqueeze(1)

    # 3. Denoising loop
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
        noise_pred = unet(latents, t, encoder_hidden_states=cond_embedding).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
    images = latents
    # Images are in [-1,1] → rescale to [0,1]
    images = (images.clamp(-1, 1) + 1) / 2
    return images

samples = sample_from_params(test_parameters)

samples = samples.cpu().numpy()
num_samples = 9   # how many test cases to generate

def plot_images_grid(images, input_parameters, save_path):
    fig, axes = plt.subplots(int(np.sqrt(len(images))), int(np.sqrt(len(images))), figsize=(12,12))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(images[i].squeeze(), cmap='RdBu_r')
        ax.set_title(f"Alpha1={input_parameters[i][0].item():.2f}, Alpha2={input_parameters[i][1].item():.2f}")
        plt.colorbar(im, ax=ax)
        ax.axis('off')
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

plot_images_grid(
    samples[:num_samples],
    test_parameters[:num_samples],
    f"test_predictions.png"
)

plot_images_grid(
    test_data[:num_samples],
    test_parameters[:num_samples],
    f"test_true_values.png"
)

plt.figure(figsize=(12, 6))
for i in range(num_samples):
    plt.subplot(2, num_samples // 2, i + 1)
    plt.imshow(samples[i, 0], cmap="viridis")  # assuming grayscale
    plt.axis("off")
plt.suptitle("Generated Test Predictions", fontsize=16)
plt.tight_layout()
plt.savefig("test_predictions.png")
plt.show()

print(f"MSE: {((test_data - samples)**2).mean()}")