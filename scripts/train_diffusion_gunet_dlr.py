import os
import shutil
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph
from tqdm import tqdm
from einops import rearrange

from denoising_diffusion_pytorch.graph_unet_cfg_diffusion import (
    ConditionedGraphUNet,
    GraphDiffusion,
    Trainer,
)
import pyLOM
from cetaceo.evaluators import RegressionEvaluator


data_dir = "/home/d.ramos/Datos_DLR_pylom"

def load_dataset(name, norm_coefficients):
    data_train = pyLOM.Dataset.load(f"{data_dir}/{name}.h5")
    airfoil_coords = torch.tensor(data_train.xyz).float()
    aoa = data_train.get_variable('AoA')
    mach = data_train.get_variable('Mach')
    cp = data_train["CP"].T
    if "cp_min" not in norm_coefficients:
        norm_coefficients["cp_min"] = cp.min()
        norm_coefficients["cp_max"] = cp.max()
    cp = (cp - norm_coefficients["cp_min"]) / (norm_coefficients["cp_max"] - norm_coefficients["cp_min"])

    if "mach_mean" not in norm_coefficients:
        norm_coefficients["mach_mean"] = mach.mean()
        norm_coefficients["mach_std"] = mach.std()
    mach = (mach - norm_coefficients["mach_mean"]) / norm_coefficients["mach_std"]

    if "aoa_mean" not in norm_coefficients:
        norm_coefficients["aoa_mean"] = aoa.mean()
        norm_coefficients["aoa_std"] = aoa.std()
    aoa = (aoa - norm_coefficients["aoa_mean"]) / norm_coefficients["aoa_std"]

    cp_t = torch.from_numpy(cp).float()
    aoa_t = torch.from_numpy(aoa).float()
    mach_t = torch.from_numpy(mach).float()
    print(cp.min(), cp.max())
    print(aoa.mean(), aoa.std())
    print(mach.mean(), mach.std())

    return TensorDataset(cp_t, aoa_t, mach_t), airfoil_coords

def collate_fn(batch, mesh_coords, edge_index):
    aoas = torch.tensor([data[1] for data in batch])
    machs = torch.tensor([data[2] for data in batch])
    conditions = torch.stack((machs, aoas), dim=1)
    graphs = [
        Data(
            x=cp.reshape(-1, 1),
            pos=mesh_coords,
            edge_index=edge_index.clone(),
        )
        for cp, aoa, mach in batch
    ]
    return Batch.from_data_list(graphs), conditions
    # batched = Batch.from_data_list(graphs)
    # # Compute edge_index for the entire batch
    # batched.edge_index = knn_graph(batched.pos, k=2, batch=batched.batch, loop=False)
    # return batched

norm_coefficients = {}

dataset_train, airfoil_coords = load_dataset("NRL7301_TRAIN", norm_coefficients)
# compute mesh connectivity
edge_index = knn_graph(airfoil_coords, k=2, batch=None, loop=False)
print(airfoil_coords.min(axis=0), airfoil_coords.max(axis=0))
num_points = airfoil_coords.shape[0]
# compute the connectivity
# edge_index = knn_graph(airfoil_coords, k=2, batch=None, loop=False)
dataset_test, _ = load_dataset("NRL7301_TEST", norm_coefficients)
example_graph = dataset_train[0][0]

model = ConditionedGraphUNet(
    dim=64,
    in_channels=1,
    out_channels=1,
    cond_dim=2,
    cond_drop_prob=0.0,
    dim_mults=(1, 2, 4),
    pool_ratios=0.5,
    sum_res=False,
    act=torch.nn.GELU(),
)


diffusion = GraphDiffusion(
    model,
    num_mesh_points=example_graph.shape[0],
    default_mesh_connectivity=edge_index,
    objective="pred_noise",  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=500,
    timesteps=500,  # number of steps
    min_snr_loss_weight=True,
    min_snr_gamma=5,
)

results_folder = "results/dlr/gunet_M"

train_steps = 10000

trainer = Trainer(
    diffusion,
    dataset=dataset_train,
    train_batch_size=13,
    train_lr=1e-5,
    num_samples=9,
    train_num_steps=train_steps+4,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.95,  # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=12000,
    # use_cpu=True,
    dl_collate_fn=partial(collate_fn, mesh_coords=airfoil_coords, edge_index=edge_index)
)

trainer.train()