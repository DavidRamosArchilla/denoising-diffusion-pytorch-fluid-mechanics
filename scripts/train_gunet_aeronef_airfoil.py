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
from cetaceo.evaluators import RegressionEvaluator

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True  

def get_split_indices(split_name, split_data, all_conds, atol=1e-4, rtol=1e-12):
    data = split_data[split_name]
    indices = []

    for row in data:
        # Compare this row against all rows in all_conds
        matches = np.all(np.isclose(all_conds, row, atol=atol, rtol=rtol), axis=1)
        found = np.where(matches)[0]

        if len(found) == 0:
            raise KeyError(f"No match found within tolerance for {row}")
        if len(found) > 1:
            raise ValueError(f"Multiple matches found within tolerance for {row}")

        indices.append(found[0])

    return indices

def sanity_check_splits(split_data, all_conds):
    train_idx = set(get_split_indices("Train", split_data, all_conds=all_conds))
    val_idx   = set(get_split_indices("Validation", split_data, all_conds=all_conds))
    test_idx  = set(get_split_indices("Test", split_data, all_conds=all_conds))

    # Check pairwise disjointness
    assert train_idx.isdisjoint(val_idx),  "Train and Validation overlap!"
    assert train_idx.isdisjoint(test_idx), "Train and Test overlap!"
    assert val_idx.isdisjoint(test_idx),   "Validation and Test overlap!"

    # Optional: check coverage
    all_idx = train_idx | val_idx | test_idx
    assert len(all_idx) == len(split_data["All"]), \
        "Splits do not cover all samples exactly once"

    print("✅ Sanity check passed: splits are disjoint and complete.")

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

data = np.load("data/aeronef/db_random.npy", allow_pickle=True).item()
field_name_to_predict = "Cp" # "Pressure", Cp
# cp = data["Cp"]
# train_size = 0.8
# training_indices = np.random.choice(cp.shape[0], int(cp.shape[0] * train_size), replace=False)
# test_indices = np.setdiff1d(np.arange(cp.shape[0]), training_indices)
split_data = np.load("data/aeronef/best_train-val-test_split.npy", allow_pickle=True).item()
alpha, vel_inf = data["Alpha"], data["Vinf"]
all_conds = np.stack((alpha, vel_inf), axis=1)
training_indices = get_split_indices("Train", split_data, all_conds=all_conds)
val_indices = get_split_indices("Validation", split_data, all_conds=all_conds)
test_indices = get_split_indices("Test", split_data, all_conds=all_conds)
sanity_check_splits(split_data, all_conds=all_conds)
# import sys;sys.exit()

def load_dataset(indices, norm_coefficients, data):
    aoa = data["Alpha"][indices]
    vinf = data["Vinf"][indices]
    cp = data[field_name_to_predict][indices]
    # if "cp_min" not in norm_coefficients:
    #     norm_coefficients["cp_min"] = cp.min()
    #     norm_coefficients["cp_max"] = cp.max()
    # cp = (cp - norm_coefficients["cp_min"]) / (norm_coefficients["cp_max"] - norm_coefficients["cp_min"])
    # feature_range = [-1, 1]
    # cp = cp * (feature_range[1] - feature_range[0]) + feature_range[0]
    # norm_coefficients["feature_range"] = feature_range
    if "cp_mean" not in norm_coefficients:
        norm_coefficients["cp_mean"] = cp.mean()
        norm_coefficients["cp_std"] = cp.std()
    cp = (cp - norm_coefficients["cp_mean"]) / norm_coefficients["cp_std"]

    if "vinf_mean" not in norm_coefficients:
        norm_coefficients["vinf_mean"] = vinf.mean()
        norm_coefficients["vinf_std"] = vinf.std()
    vinf = (vinf - norm_coefficients["vinf_mean"]) / norm_coefficients["vinf_std"]

    if "aoa_mean" not in norm_coefficients:
        norm_coefficients["aoa_mean"] = aoa.mean()
        norm_coefficients["aoa_std"] = aoa.std()
    aoa = (aoa - norm_coefficients["aoa_mean"]) / norm_coefficients["aoa_std"]

    # pad/truncate to length 27500
    # target_len = 27500 # 704 = 64*11 | 27499 = 257*107
    # if field_name_to_predict == "Pressure" and cp.shape[1] < target_len:
    #     pad_width = target_len - cp.shape[1]
    #     cp = np.pad(cp, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    # pad_width = 696 - cp.shape[1]
    # cp = np.pad(cp, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    cp_t = torch.from_numpy(cp).float()
    # cp_t = cp_t[..., :-1]
    aoa_t = torch.from_numpy(aoa).float()
    mach_t = torch.from_numpy(vinf).float()
    conditions = torch.stack([aoa_t, mach_t], dim=1)
    print(cp.mean(), cp.std())
    print(aoa.mean(), aoa.std())
    print(vinf.mean(), vinf.std())
    print(cp_t.shape, conditions.shape)

    return TensorDataset(cp_t, aoa_t, mach_t)

norm_coefficients = {}
dataset_train = load_dataset(training_indices, norm_coefficients, data)
val_dataset = load_dataset(val_indices, norm_coefficients, data)
dataset_test = load_dataset(test_indices, norm_coefficients, data)
airfoil_coords = torch.tensor(data["Airfoil"]).float()
airfoil_coords_mean = airfoil_coords.mean(axis=0, keepdims=True)
airfoil_coords_std = airfoil_coords.std(axis=0, keepdims=True)
airfoil_coords = (airfoil_coords - airfoil_coords_mean) / airfoil_coords_std

num_points = airfoil_coords.shape[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConditionedGraphUNet(
    dim=128,
    in_channels=4,
    out_channels=1,
    # cond_dim=2,
    cond_drop_prob=0.0,
    dim_mults=(1, 2, 4),
    attention_layers=[],
    pool_ratios=0.5,
    sum_res=True,
    act=torch.nn.GELU(),
    attn_heads=8,
    attn_dim_head=64
)
model.to(device)
# model = torch.compile(model)
results_dir = Path("results/aeronef_cp_good_split/solo_gunet_NOSE_WITHOUT_attn_sum_res_scale_coords_w_scheduler")
os.makedirs(results_dir, exist_ok=True)

training_iters = 60000
batch_size = 32
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
    # unscale predictions and targets
    predictions_np = predictions_np * norm_coefficients["cp_std"] + norm_coefficients["cp_mean"]
    targets_np = targets_np * norm_coefficients["cp_std"] + norm_coefficients["cp_mean"]
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_iters, eta_min=1e-6)

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


    
