import numpy as np
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D, Trainer1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.learned_gaussian_diffusion import LearnedGaussianDiffusion1D
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching

import shutil
import os
import matplotlib.pyplot as plt
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
dataset_test = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_parameters, dtype=torch.float32))
model = Unet1D(
    dim = 64,
    dim_mults=(1, 2, 4), #, 8),
    # flash_attn = False,
    channels=2,
    cond_dim=2,
    # attn_heads=8,
    # full_attn=True,
    # cross_attn=True,
    learned_sinusoidal_cond=True,
    learn_sigma=False,
    # self_condition=True
)

# diffusion = GaussianDiffusion1D(
#     model,
#     seq_length=data.shape[-1], # 298 points of the airfoil + 2 padding --> 300 (which is divisible by 4)
#     objective='pred_noise',  # 'pred_noise' or 'pred_x0'
#     beta_schedule="cosine",
#     sampling_timesteps=1000,
#     timesteps=1000,    # number of steps
#     # use_cfg_plus_plus=False,
#     min_snr_loss_weight=True,
#     min_snr_gamma=5
# )

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=data.shape[-1],
    cond_scale=6,
    num_sampling_steps=500,
    sampling_method="euler",
)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 2**30} GB")

results_folder = 'results/dlr/unet_big_FM_euler_500'
# results_folder = 'results/dlr/test'


trainer = Trainer1D(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset,
    dataset_test=dataset_test,
    train_batch_size=8,
    train_lr=1e-4,
    num_samples=9,
    train_num_steps=100001,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.99,  # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=10000,
    max_grad_norm=1.0,
    # use_cpu=True,
    # compile_model=True
)
torch.cuda.empty_cache()
# trainer.load(7)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
trainer.train()
# trainer.load(10)
trainer.ema.ema_model.eval() 
# diffusion = trainer.accelerator.unwrap_model(diffusion)
diffusion.eval()

errors, samples = evaluate_model(
    trainer.ema.ema_model, # trainer.ema.ema_model, # diffusion
    test_parameters,
    test_data,
    32,
    cond_scale=6
)
print(f"Final errors:\n{errors}")
torch.save(samples, f"{results_folder}/test_predictions_ema.pt")

# denoising_steps = 500
# model_device = next(diffusion.parameters()).device
# initial_input_train = data[34]
# initial_input_train = torch.tensor(initial_input_train, device=model_device).float().unsqueeze(0)
# interpolation_input = torch.tensor(data[9], device=model_device).float().unsqueeze(0)
# interpolation_parameters = torch.tensor(parameters[34] + parameters[9], device=model_device, dtype=torch.float32).unsqueeze(0).mean(dim=0).unsqueeze(0)
# # print(interpolation_parameters.shape, (parameters[34] + parameters[9]).shape)
# interpolation_pred = diffusion.interpolate(initial_input_train, interpolation_input, classes=interpolation_parameters, t=denoising_steps)

# target_test_idx = 16
# test_parameters = torch.tensor(test_parameters[target_test_idx], device=model_device).float().unsqueeze(0)
# test_solution = torch.tensor(test_data[target_test_idx]).float()

# pred = diffusion.sample(classes=test_parameters, cond_scale=6).cpu()

# img_to_img_pred = diffusion.img_to_img(initial_input_train, test_parameters, t=denoising_steps).cpu()
# img_to_img_with_interpolation = diffusion.img_to_img(interpolation_pred, test_parameters, t=denoising_steps).cpu()

# base_mae = torch.abs(pred - test_solution).mean()
# img_to_img_mae = torch.abs(img_to_img_pred - test_solution).mean()
# train_test_mae = torch.abs(initial_input_train.cpu() - test_solution).mean()
# interpolation_mae = torch.abs(interpolation_pred.cpu() - test_solution).mean()
# img_to_img_with_interpolation_mae = torch.abs(img_to_img_with_interpolation - test_solution).mean()

# print(f"MAE: {base_mae}  With img to img: {img_to_img_mae}  MAE if train sample was the prediction {train_test_mae}  interpolation MAE: {interpolation_mae}  el ultimo: {img_to_img_with_interpolation_mae}")

# from netCDF4 import Dataset
# fname = "data/dlr_airfoils/test/Snap_Case0060_M0.63047_AoA1.64198"
# nc_fid = Dataset(fname, 'r')
# x = nc_fid.variables["x"][: 597]
# cp_pred = 