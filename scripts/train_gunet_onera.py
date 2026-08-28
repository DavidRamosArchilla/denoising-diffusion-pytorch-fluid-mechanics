import os
import shutil
from functools import partial
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
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
from cetaceo.evaluators import RegressionEvaluator

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True  

def collate_fn(batch, mesh_coords, edge_index):
    graphs = [
        Data(
            x=torch.cat(
                [mesh_coords, torch.full_like(mesh_coords[:, 0:1], mach.item()), torch.full_like(mesh_coords[:, 0:1], aoa.item()), torch.full_like(mesh_coords[:, 0:1], p_i.item())], dim=1,
            ),
            pos=mesh_coords,
            y=cp,
            edge_index=edge_index.clone(),
        )
        for cp, aoa, mach, p_i in batch
    ]
    return Batch.from_data_list(graphs)
    # batched = Batch.from_data_list(graphs)
    # # Compute edge_index for the entire batch
    # batched.edge_index = knn_graph(batched.pos, k=2, batch=batched.batch, loop=False)
    # return batched

qoi_list = ['cp', 'cfx', 'cfy', 'cfz'] # names of the quantites of interest
nwallp = 260774  # number of points on the aircraft skin

data_dir = "/home/airbus/onera_data"
X_train_tot = np.load(data_dir + '/X_train.npy')
Y_train_tot = np.load(data_dir + '/Ytrain.npy')
X_test = np.load(data_dir + '/X_test.npy')
Y_test = np.load(data_dir + '/Ytest.npy')

df_description = pd.read_csv(data_dir + '/describe_train_test_repartition_with_weights.csv', index_col=0)
df_test = df_description.loc[~df_description['Train']]
df_train = df_description.loc[df_description['Train']]

ncase = len(df_description)  # 468
ntest = len(df_test) # 156
ntrain = ncase-ntest  # 312

# extract xyz coordinates 
airfoil_coords = torch.tensor(X_train_tot[0:nwallp,:3], dtype=torch.float32)

airfoil_coords_mean = airfoil_coords.mean(dim=0, keepdim=True)
airfoil_coords_std = airfoil_coords.std(dim=0, keepdim=True)
airfoil_coords = (airfoil_coords - airfoil_coords_mean) / airfoil_coords_std

# Remove the geometric informations from the input array
X_train_tot_conditions = X_train_tot[0::nwallp,6:9]
X_test_conditions = X_test[0::nwallp,6:9]
# Create the output array to be of shape (ntrain, nwallp, 4)
Y_train_tot_conditions = np.array([Y_train_tot[nwallp*i:nwallp*(i+1),:] for i in range(ntrain)])
Y_test_tot_conditions = np.array([Y_test[nwallp*i:nwallp*(i+1),:] for i in range(ntest)])

print("X_train_tot_conditions shape", X_train_tot_conditions.shape)
print("Y_train_tot_conditions shape", Y_train_tot_conditions.shape)
# split X_train_tot and Y_train_tot into train and validation arrays
X_train = X_train_tot_conditions
X_test = X_test_conditions

Y_train = Y_train_tot_conditions
Y_test = Y_test_tot_conditions

# process with torch 
# first, move channels dim
Y_train = torch.tensor(Y_train, dtype=torch.float32).permute(0, 2, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).permute(0, 2, 1)

# normalize/standarize things
train_mean = Y_train.mean(dim=(0, 2), keepdim=True)
train_std  = Y_train.std(dim=(0, 2), keepdim=True)
condition_mean = X_train.mean(axis=0, keepdims=True)
condition_std = X_train.std(axis=0, keepdims=True) 

Y_train = (Y_train - train_mean) / train_std
Y_test = (Y_test - train_mean) / train_std
X_train = (X_train - condition_mean) / condition_std
X_test = (X_test - condition_mean) / condition_std

# pad sequences to a multple of a power of 2. 260864 = 256 * 1019
original_length = Y_train.shape[2]
# pad_length = 260864
# Y_train = F.pad(Y_train, (0, pad_length - nwallp))
# Y_test = F.pad(Y_test, (0, pad_length - nwallp))

print("X train shape", X_train.shape)
print("X test shape", X_test.shape)
print("Y train shape", Y_train.shape)
print("Y test shape", Y_test.shape)
print("mean/std X train test ", X_train.mean(axis=0), X_test.mean(axis=0), X_train.std(axis=0), X_test.std(axis=0))
print("mean/std Y train test ", Y_train.mean(dim=(0, 2)), Y_test.mean(dim=(0, 2)), Y_train.std(dim=(0, 2)), Y_test.std(dim=(0, 2)))
dataset_train = TensorDataset(
    Y_train, torch.tensor(X_train[:, 0:1], dtype=torch.float32), torch.tensor(X_train[:, 1:2], dtype=torch.float32), torch.tensor(X_train[:, 2:], dtype=torch.float32)
)

dataset_test = TensorDataset(
    Y_test, torch.tensor(X_test[:, 0:1], dtype=torch.float32), torch.tensor(X_test[:, 1:2], dtype=torch.float32), torch.tensor(X_test[:, 2:], dtype=torch.float32)
)

norm_coefficients = {
    "cp_mean": train_mean.squeeze().cpu().numpy(),
    "cp_std": train_std.squeeze().cpu().numpy(),
    "condition_mean": condition_mean.squeeze(),
    "condition_std": condition_std.squeeze(),
}
num_points = airfoil_coords.shape[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
out_channels = 4
model = ConditionedGraphUNet(
    dim=128,
    in_channels=6, # xyz, aoa, m, pi
    out_channels=out_channels,
    # cond_dim=2,
    cond_drop_prob=0.0,
    attention_layers=[1, 2],
    dim_mults=(1, 2, 4),
    pool_ratios=0.5,
    sum_res=True,
    act=torch.nn.GELU(),
    attn_heads=8,
    attn_dim_head=32
)
model.to(device)
# model = torch.compile(model)
results_dir = Path("results/onera/solo_gunet_NOSE_with_attn_sum_res_scheduler")
os.makedirs(results_dir, exist_ok=True)

training_iters = 30000
batch_size = 1
lr = 1e-4

def cycle(dl):
    while True:
        for data in dl:
            yield data

def eval_model(model, dl_test):
    model.eval()
    with torch.inference_mode():
        preds_list = []
        targets_list = []
        for data in dl_test:
            inputs, edge_index, targets = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            inputs = rearrange(inputs, "(b n) c -> b c n", n=num_points)
            targets = rearrange(targets, "(b c) n -> b c n", c=out_channels)
            outputs = model(inputs, edge_index)  # (b, c, n)
            outputs = outputs.squeeze(1)  # (b, n)
            preds_list.append(outputs.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())
            # Clear GPU cache between batches
            del inputs, edge_index, targets, outputs
            torch.cuda.empty_cache()
        predictions_np = np.concatenate(preds_list, axis=0)
        targets_np = np.concatenate(targets_list, axis=0)
        # print("predictions shape:", predictions_np.shape, "targets shape:", targets_np.shape)
    model.train()
    # save model and optimizer state dicts as ckeckpoints
    # unscale predictions and targets
    predictions_np = predictions_np * norm_coefficients["cp_std"][None, :, None] + norm_coefficients["cp_mean"][None, :, None]
    targets_np = targets_np * norm_coefficients["cp_std"][None, :, None] + norm_coefficients["cp_mean"][None, :, None]
    return predictions_np, targets_np

if __name__ == "__main__":
    edge_index = knn_graph(airfoil_coords, k=6, batch=None, loop=False)
    dl = DataLoader(dataset_train, batch_size=batch_size, collate_fn=partial(collate_fn, mesh_coords=airfoil_coords, edge_index=edge_index))
    dl_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=partial(collate_fn, mesh_coords=airfoil_coords, edge_index=edge_index))
    dl = cycle(dl)
    test = next(iter(dl))
    print(test)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, fused=True)
    # cosine annealing lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_iters, eta_min=1e-6)

    steps = 0
    all_losses = []
    test_losses = []
    eval_test_every = 5000
    with tqdm(total=training_iters) as pbar:
        while steps < training_iters:
            data = next(dl)
            inputs, edge_index, targets = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            inputs = rearrange(inputs, "(b n) c -> b c n", n=num_points)
            targets = rearrange(targets, "(b c) n -> b c n", c=out_channels)
            # print(f"edge_index max: {edge_index.max()}, expected max: {inputs.shape[0] * inputs.shape[2] - 1}")
            # print(f"Batch: inputs shape: {inputs.shape}, edge_index shape: {edge_index.shape}")
            preds = model(inputs, edge_index)
            # (b, c, n) -> (b, n), since c = 1
            preds = preds.squeeze(1)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            steps += 1
            all_losses.append(loss.item())
            pbar.set_description(f'loss: {loss.item():.5f}')
            pbar.update(1)

            if steps % eval_test_every == 0:
                predictions_np, targets_np = eval_model(model, dl_test)
                test_loss = ((predictions_np - targets_np) ** 2).mean()
                test_losses.append(test_loss)
                torch.save(model.state_dict(), results_dir / "gunet_state_dict_ckp.pt")
                torch.save(optimizer.state_dict(), results_dir / "optimizer_state_dict_ckp.pt")
 
    # torch.cuda.empty_cache() 
    # model.load_state_dict(torch.load(results_dir / "gunet_state_dict.pt", map_location=device))

    evaluator = RegressionEvaluator(tolerance=1e-5)
    metrics = evaluator(targets_np, predictions_np)
    evaluator.print_metrics()
    plt.figure()
    plt.plot(all_losses, label='Loss')
    test_x_values = list(range(eval_test_every, training_iters+1, eval_test_every))
    plt.plot(test_x_values, test_losses, label='Test Loss')
    # Compute moving average
    window_size = 100
    if len(all_losses) >= window_size:
        moving_avg = np.convolve(all_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(all_losses)), moving_avg, label=f'Moving Avg ({window_size})')
    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss Evolution')
    plt.legend()
    plt.savefig(results_dir / "loss_evolution.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    # Save predictions and targets
    np.savez_compressed(results_dir / "predictions.npz", predictions=predictions_np, targets=targets_np)
    # save model state dict and full model
    torch.save(model.state_dict(), results_dir / "gunet_state_dict.pt")
    # save the optimizer too to resume training later
    torch.save(optimizer.state_dict(), results_dir / "optimizer_state_dict.pt")

    # make sure values are JSON-serializable (convert numpy types to native Python floats)
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        if np.isscalar(obj) or isinstance(obj, np.generic):
            return float(obj)
        try:
            # torch tensors / numpy arrays
            return to_serializable(obj.tolist())
        except Exception:
            try:
                return float(obj)
            except Exception:
                return obj

    norm_serializable = {k: (float(v) if np.isscalar(v) or isinstance(v, np.generic) else v) for k, v in norm_coefficients.items()}
    with open(results_dir / "norm_coefficients.json", "w") as f:
        json.dump(norm_serializable, f, indent=2)

    # Save metrics as JSON (make sure all values are JSON-serializable)
    metrics_serializable = to_serializable(metrics)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics_serializable, f, indent=2)


    
