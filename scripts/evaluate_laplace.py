import numpy as np
import torch
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import Unet, GaussianDiffusion, evaluate_model, Trainer
import matplotlib.pyplot as plt


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
    channels = 1,
    cond_dim=2,
)

diffusion = GaussianDiffusion(
    model,
    image_size = (64, 64),
    objective = 'pred_noise',  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=50,
    timesteps = 1000,    # number of steps
)

trained_model_dir = "results/laplace_conditioned_small_normalized"

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
    results_folder = trained_model_dir,  # folder to save results to
    save_and_sample_every=2000,
    augment_horizontal_flip=False,
    # use_cpu=True
)

trainer.load(5)

errors, samples = evaluate_model(
    diffusion,
    parameters,
    data,
    32,
)
print(f"Final errors:\n{errors}")
torch.save(samples, f"{trained_model_dir}/test_predictions.pt")

def plot_images_grid(images, input_parameters, save_path):
    fig, axes = plt.subplots(int(np.sqrt(len(images))), int(np.sqrt(len(images))), figsize=(12,12))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(images[i].squeeze())
        ax.set_title(f"Alpha1={input_parameters[i][0].item():.2f}, Alpha2={input_parameters[i][1].item():.2f}")
        plt.colorbar(im, ax=ax)
        ax.axis('off')
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

model_device = next(diffusion.parameters()).device
num_images = 9
inputs_to_plot = parameters[:num_images]
real_values_to_plot = data[:num_images]
predictions_to_plot = diffusion.sample(torch.tensor(inputs_to_plot, dtype=torch.float32).to(model_device)).cpu().numpy()

plot_images_grid(
    predictions_to_plot,
    inputs_to_plot,
    f"{trained_model_dir}/test_predictions.png"
)

plot_images_grid(
    real_values_to_plot,
    inputs_to_plot,
    f"{trained_model_dir}/test_true_values.png"
)
plt.close()