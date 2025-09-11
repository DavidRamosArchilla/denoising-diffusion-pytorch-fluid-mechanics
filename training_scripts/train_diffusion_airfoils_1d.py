
import os

import aerosandbox as asb
from aerosandbox import _asb_root
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D

import matplotlib.pyplot as plt


UIUC_DIR = _asb_root / "geometry" / "airfoil" / "airfoil_database"
 
airfoil_names = os.listdir(UIUC_DIR) 
airfoil_names.remove("utils")
n_points_per_side = 128 #128 100
airfoil_points = list(map(lambda airfoil_name: asb.Airfoil(airfoil_name), airfoil_names))

sample_airfoil = airfoil_points[0].repanel(n_points_per_side=n_points_per_side)
x_coords_upper = sample_airfoil.upper_coordinates()[:, 0]
x_coords_lower = sample_airfoil.lower_coordinates()[:, 0]

def airfoil_to_points(airfoil: asb.Airfoil):
    airfoil = airfoil.repanel(n_points_per_side=n_points_per_side)
    y_upper = torch.tensor(airfoil.upper_coordinates()[:, 1])
    y_lower = torch.tensor(airfoil.lower_coordinates()[:, 1])
    if y_upper.shape[0] != y_lower.shape[0]:
        return torch.tensor([])    
    return torch.vstack([y_upper, y_lower])

airfoil_points = list(map(lambda airfoil: airfoil_to_points(airfoil), airfoil_points))
airfoil_points = list(filter(lambda x: x.shape[0] > 0, airfoil_points))  # filter out empty airfoils
print(airfoil_points[0].shape)
airfoil_points = torch.stack(airfoil_points, dim=0)
airfoil_points.shape,

# create train and test datasets
train_percentage = 0.8  # You can change this value as needed
split_idx = int(train_percentage * len(airfoil_points))
train_dataset = TensorDataset(airfoil_points[:split_idx])
test_dataset = TensorDataset(airfoil_points[split_idx:]) # useless for now


def print_sample(sample_airfoil, n_points_per_side=100):
    sample_airfoil = sample_airfoil[:, :n_points_per_side]#.unsqueeze(0)
    print(sample_airfoil.shape)

    y_upper = sample_airfoil[0, :n_points_per_side]
    y_lower = sample_airfoil[1, :n_points_per_side]
    print(y_upper.shape, y_lower.shape)
    plt.scatter(x_coords_upper, y_upper, marker='.')
    plt.scatter(x_coords_lower, y_lower, marker='.')
    plt.axis("equal")
    plt.show()



model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 2,  # 2 for upper and lower coordinates
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 1000,
    objective = 'pred_v' # 'pred_noise' would be worth trying too
)

train_dataset = Dataset1D(airfoil_points[:split_idx].float())
test_dataset = Dataset1D(airfoil_points[split_idx:].float())

experiment_name = "diffusion_airfoils_1d"
trainer = Trainer1D(
    diffusion,
    dataset = train_dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    results_folder=f"./results/{experiment_name}",
)

trainer.train()
