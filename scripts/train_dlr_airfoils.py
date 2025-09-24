import numpy as np
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import Unet1D, GaussianDiffusion1D, Trainer1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model

import shutil
import os
import matplotlib.pyplot as plt
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Get details about each GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# Get current device
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Current device name: {torch.cuda.get_device_name()}")


def show_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Multi-processors: {props.multi_processor_count}")

show_gpu_info()

data = np.load("data/dlr_airfoils/cp_train.npy")
# at the moment the model expects inputs in [0, 1], like grayscale images 
data_min, data_max = data.min(), data.max()
data = (data - data_min) / (data_max - data_min)  # scale to [0, 1]
# pad the sequences so they are divisible by 4 and i have no problems with the downsampling of the unet
pad_width = ((0, 0),    # No padding for the N samples dimension
             (0, 0),    # No padding for the 2 channels dimension
             (1, 1))    # Add 1 element before and 1 after the sequence dimension
# data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)

parameters = np.load("data/dlr_airfoils/conditions_train.npy")
parameters_mean, parameters_std = parameters.mean(axis=0), parameters.std(axis=0)
parameters = (parameters - parameters_mean) / (parameters_std)

# load test data
test_data = np.load("data/dlr_airfoils/cp_test.npy")
test_data = (test_data - data_min) / (data_max - data_min) 
# test_data = np.pad(test_data, pad_width=pad_width, mode='constant', constant_values=0)
test_parameters = np.load("data/dlr_airfoils/conditions_test.npy")
test_parameters = (test_parameters - parameters_mean) / (parameters_std)

dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(parameters, dtype=torch.float32))

model = Unet1D(
    dim = 64,
    dim_mults=(1, 2, 4, 8),
    # flash_attn = False,
    channels=2,
    cond_dim=2,
    # full_attn = False
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=298, # 298 points of the airfoil + 2 padding --> 300 (which is divisible by 4)
    objective='pred_noise',  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=1000,
    timesteps=1000,    # number of steps
    # use_cfg_plus_plus=False
)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 1e9} GB")

results_folder = 'results/dlr_big'

trainer = Trainer1D(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset,
    train_batch_size=16,
    train_lr=8e-5,
    num_samples=9,
    train_num_steps=30004,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=2500,
    # use_cpu=True
)

shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
# trainer.load(1)
trainer.train()
# trainer.load(5)
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
