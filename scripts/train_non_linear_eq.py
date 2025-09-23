import numpy as np
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import Unet, GaussianDiffusion, Trainer, evaluate_model
import shutil
import os
import matplotlib.pyplot as plt
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



data = np.load("data/non_linear_eq/non_linear_train_solutions.npy")
# at the moment the model expects inputs in [0, 1], like grayscale images 
data_min, data_max = data.min(), data.max()
data = (data - data_min) / (data_max - data_min)  # scale to [0, 1]
data = data[:, np.newaxis, :, :]  # add channel dimension   
parameters = np.load("data/non_linear_eq/non_linear_train_parameters.npy")
# normalize parameters to [0, 1]
parameters_mean, parameters_std = parameters.mean(axis=0), parameters.std(axis=0)
parameters = (parameters - parameters_mean) / (parameters_std)

# load test data
test_data = np.load("data/non_linear_eq/non_linear_test_solutions.npy")
test_data = (test_data - data_min) / (data_max - data_min) 
test_data = test_data[:, np.newaxis, :, :]
test_parameters = np.load("data/non_linear_eq/non_linear_test_parameters.npy")
test_parameters = (test_parameters - parameters_mean) / (parameters_std)

dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(parameters, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_parameters, dtype=torch.float32))

model = Unet(
    dim = 128,
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
    sampling_timesteps=1000,
    timesteps=1000,    # number of steps
    # use_cfg_plus_plus=True
)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 1e9} GB")

results_folder = 'results/non_linear_eq_big_128_bs'

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
    # amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    results_folder = results_folder,  # folder to save results to
    save_and_sample_every=2000,
    augment_horizontal_flip=False,
    # use_cpu=True
)

# save this script in the results folder for reference
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
# trainer.load(1)
trainer.train()
# trainer.load(5)
# torch.cuda.empty_cache()  # Clear GPU memory
# trainer.ema.ema_model.eval()  # Ensure eval mode
diffusion = trainer.accelerator.unwrap_model(diffusion)
diffusion.eval()

errors, samples = evaluate_model(
    diffusion, # trainer.ema.ema_model, #
    test_parameters,
    test_data,
    32,
    cond_scale=6
)
print(f"Final errors:\n{errors}")
torch.save(samples, f"{results_folder}/test_predictions.pt")

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
inputs_to_plot = test_parameters[:num_images]
real_values_to_plot = test_data[:num_images]
predictions_to_plot = diffusion.sample(torch.tensor(inputs_to_plot, dtype=torch.float32).to(model_device)).cpu().numpy()

plot_images_grid(
    predictions_to_plot,
    inputs_to_plot,
    f"{results_folder}/test_predictions.png"
)

plot_images_grid(
    real_values_to_plot,
    inputs_to_plot,
    f"{results_folder}/test_true_values.png"
)
plt.close()