import numpy as np
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import Unet, GaussianDiffusion, Trainer, evaluate_model
from denoising_diffusion_pytorch.karras_unet import KarrasUnet
from denoising_diffusion_pytorch.dit import DiT_models, DiT
import shutil
import os
import matplotlib.pyplot as plt
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



data = np.load("data/non_linear_eq/non_linear_train_solutions.npy")
# data = np.load("data/non_linear_eq_latents/train_latents.npy") 
# at the moment the model expects inputs in [0, 1], like grayscale images 
data_min, data_max = data.min(), data.max()
data = (data - data_min) / (data_max - data_min)  # scale to [0, 1]
data = data[:, np.newaxis, :, :]  # add channel dimension   
print(data.shape)
parameters = np.load("data/non_linear_eq/non_linear_train_parameters.npy")
# parameters = np.load("data/non_linear_eq_latents/train_parameters.npy") 
# normalize parameters to [0, 1]
parameters_mean, parameters_std = parameters.mean(axis=0), parameters.std(axis=0)
parameters = (parameters - parameters_mean) / (parameters_std)

# load test data
test_data = np.load("data/non_linear_eq/non_linear_test_solutions.npy")
test_data = (test_data - data_min) / (data_max - data_min) 
test_data = test_data[:, np.newaxis, :, :]
test_parameters = np.load("data/non_linear_eq/non_linear_test_parameters.npy")
test_parameters = (test_parameters - parameters_mean) / (parameters_std)

extrapolation_data = np.load("data/non_linear_eq/non_linear_extrapolate_solutions.npy")
extrapolation_data = (extrapolation_data - data_min) / (data_max - data_min) 
extrapolation_data = extrapolation_data[:, np.newaxis, :, :]

extrapolation_parameters = np.load("data/non_linear_eq/non_linear_extrapolate_parameters.npy")
extrapolation_parameters = (extrapolation_parameters - parameters_mean) / (parameters_std)

dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(parameters, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_parameters, dtype=torch.float32))
# (B, 1, 64, 64), (B, 2)

# para usar la unet
model = Unet(
    dim = 128,
    dim_mults = (1, 2, 4),#, 8),
    # flash_attn = False,
    channels = 1, # 4 for the latent representations
    cond_dim=2,
    # full_attn = False
)

# para usar el transformer
# model = DiT(
#     depth=14,
#     hidden_size=896,
#     patch_size=1,
#     num_heads=14,
#     input_size=latents_train.shape[2],
#     cond_dim=dataset_train.tensors[1].shape[1],
#     class_dropout_prob=0.2,
#     in_channels=dataset_train.tensors[0].shape[1],
#     learn_sigma=False,
#     use_swiglu=True,
#     attn_type="linear", # linear vanilla triton_linear
#     qk_norm=True, # to avoid stability issues with bf16
#     mlp_ratio=2.5,
#     # num_experts=4,
#     # num_experts_per_tok=2
# )

diffusion = GaussianDiffusion(
    model,
    image_size = (64, 64),
    objective = 'pred_noise',  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=1000,
    timesteps=1000,    # number of steps
    # use_cfg_plus_plus=True
)
# funciona un poco mejor [Flow matching]
# sampler = Sampler(transport=create_transport(
#     # use_cosine_loss=True,
#     # use_lognorm=True
# ))

# diffusion = FlowMatching(
#     sampler,
#     model,
#     input_size=dataset.tensors[0].shape[2],
#     cond_scale=2,
#     num_sampling_steps=400,
#     sampling_method="euler",
#     # shifted_mu=1.0986
# )

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 1e9} GB")

results_folder = 'results/non_linear_eq_big_karras'

trainer = Trainer(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    num_samples=9,
    train_num_steps = 30004,         # total training steps
    gradient_accumulate_every = 4,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    # amp = True,                       # turn on mixed precision to make it faster
    # mixed_precision_type = 'bf16',
    calculate_fid = False,              # whether to calculate fid during training
    results_folder = results_folder,  # folder to save results to
    save_and_sample_every=2000,
    augment_horizontal_flip=False,
    # use_cpu=True
)

# save this script in the results folder for reference
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
# trainer.load(6)
trainer.train()
# trainer.load(15)
# torch.cuda.empty_cache()  # Clear GPU memory
# trainer.ema.ema_model.eval()  # Ensure eval mode
diffusion = trainer.accelerator.unwrap_model(diffusion)
diffusion.eval()

errors, samples = evaluate_model(
    diffusion, # trainer.ema.ema_model, #
    test_parameters,
    test_data,
    128,
    cond_scale=6
)
print(f"Final errors:\n{errors}")
torch.save(samples, f"{results_folder}/latents_predictions.pt")

def plot_images_grid(images, input_parameters, save_path):
    fig, axes = plt.subplots(int(np.sqrt(len(images))), int(np.sqrt(len(images))), figsize=(12,12))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(images[i].squeeze(), cmap='RdBu_r')
        ax.set_title(f"Alpha1={input_parameters[i][0].item():.2f}, Alpha2={input_parameters[i][1].item():.2f}")
        plt.colorbar(im, ax=ax)
        ax.axis('off')
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

model_device = next(diffusion.parameters()).device
num_images = 9
inputs_to_plot = (extrapolation_parameters[:num_images] * parameters_std) - parameters_mean
real_values_to_plot = extrapolation_data[:num_images]
predictions_to_plot = diffusion.sample(torch.tensor(inputs_to_plot, dtype=torch.float32).to(model_device)).cpu().numpy()

plot_images_grid(
    predictions_to_plot,
    inputs_to_plot,
    f"{results_folder}/extrapolation_predictions.png"
)

plot_images_grid(
    real_values_to_plot,
    inputs_to_plot,
    f"{results_folder}/extrapolation_true_values.png"
)
plt.close()