import os
import shutil
from functools import partial
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
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
    graphs = [
        Data(
            x=torch.cat(
                [mesh_coords, torch.full_like(mesh_coords[:, 0:1], mach.item()), torch.full_like(mesh_coords[:, 0:1], aoa.item())], dim=1,
            ),
            pos=mesh_coords,
            y=cp,
            edge_index=edge_index.clone(),
        )
        for cp, aoa, mach in batch
    ]
    return Batch.from_data_list(graphs)
    # batched = Batch.from_data_list(graphs)
    # # Compute edge_index for the entire batch
    # batched.edge_index = knn_graph(batched.pos, k=2, batch=batched.batch, loop=False)
    # return batched

norm_coefficients = {}

dataset_train, airfoil_coords = load_dataset("NRL7301_TRAIN", norm_coefficients)
print(airfoil_coords.min(axis=0), airfoil_coords.max(axis=0))
num_points = airfoil_coords.shape[0]
# compute the connectivity
# edge_index = knn_graph(airfoil_coords, k=2, batch=None, loop=False)
dataset_test, _ = load_dataset("NRL7301_TEST", norm_coefficients)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConditionedGraphUNet(
    dim=64,
    in_channels=4,
    out_channels=1,
    # cond_dim=2,
    cond_drop_prob=0.0,
    dim_mults=(1, 2, 4, 8),
    pool_ratios=0.5,
    sum_res=False,
    act=torch.nn.GELU(),
)
model.to(device)

results_dir = Path("results/dlr/solo_gunet_XXL_with_attn")
os.makedirs(results_dir, exist_ok=True)

training_iters = 60000
batch_size = 8
lr = 1e-4

def cycle(dl):
    while True:
        for data in dl:
            yield data

def eval_model(model, dl_test):
    model.eval()
    with torch.no_grad():
        preds_list = []
        targets_list = []
        for data in dl_test:
            inputs, edge_index, targets = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            inputs = rearrange(inputs, "(b n) c -> b c n", n=num_points)
            targets = rearrange(targets, "(b n) -> b n", n=num_points)
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
    
    return predictions_np, targets_np

if __name__ == "__main__":
    edge_index = knn_graph(airfoil_coords, k=2, batch=None, loop=False)
    dl = DataLoader(dataset_train, batch_size=batch_size, collate_fn=partial(collate_fn, mesh_coords=airfoil_coords, edge_index=edge_index))
    dl_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=partial(collate_fn, mesh_coords=airfoil_coords, edge_index=edge_index))
    dl = cycle(dl)
    test = next(iter(dl))
    print(test)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4, fused=True)
    # cosine annealing lr scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_iters, eta_min=1e-6)

    steps = 0
    all_losses = []
    test_losses = []
    eval_test_every = 500
    with tqdm(total=training_iters) as pbar:
        while steps < training_iters:
            data = next(dl)
            inputs, edge_index, targets = data.x.to(device), data.edge_index.to(device), data.y.to(device)
            inputs = rearrange(inputs, "(b n) c -> b c n", n=num_points)
            targets = rearrange(targets, "(b n) -> b n", n=num_points)
            # print(f"edge_index max: {edge_index.max()}, expected max: {inputs.shape[0] * inputs.shape[2] - 1}")
            # print(f"Batch: inputs shape: {inputs.shape}, edge_index shape: {edge_index.shape}")
            preds = model(inputs, edge_index)
            # (b, c, n) -> (b, n), since c = 1
            preds = preds.squeeze(1)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            
            steps += 1
            all_losses.append(loss.item())
            pbar.set_description(f'loss: {loss.item():.5f}')
            pbar.update(1)

            if steps % eval_test_every == 0:
                predictions_np, targets_np = eval_model(model, dl_test)
                test_loss = ((predictions_np - targets_np) ** 2).mean()
                test_losses.append(test_loss)
 
    # torch.cuda.empty_cache() 
    

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


    
