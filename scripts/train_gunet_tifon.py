"""
Train a deterministic Graph U-Net (ConditionedGraphUNet) on the TIFON dataset to
regress the surface pressure coefficient from the flow conditions (alpha, mach).

- Data loading, train/val/test splits and normalization follow scripts/train_tifon.py.
- The model / training loop follow scripts/train_gunet_onera.py: every node of the
  graph is a surface point with input features [x, z, alpha, mach] and the network
  predicts cp at that node (MSE loss, no diffusion).
"""
import pyLOM

import os
import json
import shutil
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import knn_graph
from tqdm import tqdm

from denoising_diffusion_pytorch.graph_unet_cfg_diffusion import ConditionedGraphUNet
from cetaceo.evaluators import RegressionEvaluator

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

# ----------------------------- config -----------------------------
H5_PATH = "/home/airbus/CETACEO_cp_interp/DATA/TIFON/rans5/outputs/CADGroup_3_stage_1_reordered.h5"
SPLIT_PATH = "/home/airbus/CETACEO_cp_interp/TIFON/splitting/tifon_split_complete.json"
SUBSPLITS = ["batch0", "batch1", "batch2"]
RESULTS_DIR = Path("results/tifon_final/gunet_attn_h16")

KNN_K = 2                 # neighbours per node when building the surface graph
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
TRAINING_ITERS = 30000
LR = 1e-4
ETA_MIN = 1e-6
MAX_GRAD_NORM = 1.0
EVAL_EVERY = 5000         # validation + checkpoint frequency (in steps)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- splits (same as train_tifon.py) -----------------------------
def get_split_indices(split_name, split_data, all_conds, subsplits=[], atol=1e-4, rtol=1e-12):
    if len(subsplits) > 0:
        data = []
        for subsplit_name in subsplits:
            data += split_data[split_name][subsplit_name]
    else:
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


def sanity_check_splits(split_data, all_conds, subsplits=[]):
    train_idx = set(get_split_indices("train", split_data, all_conds=all_conds, subsplits=subsplits))
    val_idx   = set(get_split_indices("validation", split_data, all_conds=all_conds, subsplits=subsplits))
    test_idx  = set(get_split_indices("test", split_data, all_conds=all_conds))

    assert train_idx.isdisjoint(val_idx),  "Train and Validation overlap!"
    assert train_idx.isdisjoint(test_idx), "Train and Test overlap!"
    assert val_idx.isdisjoint(test_idx),   "Validation and Test overlap!"

    all_idx = train_idx | val_idx | test_idx
    assert len(all_idx) == len(split_data["all"]), \
        "Splits do not cover all samples exactly once"

    print("✅ Sanity check passed: splits are disjoint and complete.")


# ----------------------------- load data -----------------------------
data = pyLOM.Dataset.load(H5_PATH)
mesh_coords = data.xyz[:, [0, 2]]                      # (N, 2) surface points (x, z)
cp = data["BoundaryValues_CoefPressure"].T             # (S, N)
cp = cp[:, np.newaxis, :]                              # (S, 1, N)
alpha, mach = torch.tensor(data.get_variable('aoa')).float(), torch.tensor(data.get_variable('mach')).float()
print(f"Loaded data: cp shape {cp.shape}, alpha shape {alpha.shape}, mach shape {mach.shape}")

with open(SPLIT_PATH) as f:
    split_data = json.load(f)

all_conds = torch.stack((alpha, mach), dim=1).numpy()
train_indices = get_split_indices("train", split_data, all_conds, SUBSPLITS)
val_indices = get_split_indices("validation", split_data, all_conds, SUBSPLITS)
test_indices = get_split_indices("test", split_data, all_conds)
sanity_check_splits(split_data, all_conds, SUBSPLITS)

train_cp, train_conds = cp[train_indices], all_conds[train_indices]
val_cp, val_conds = cp[val_indices], all_conds[val_indices]
test_cp, test_conds = cp[test_indices], all_conds[test_indices]
print(f"Split sizes: train {len(train_indices)}, val {len(val_indices)}, test {len(test_indices)}")

# normalization (statistics from the training split only)
cp_mean, cp_std = train_cp.mean(), train_cp.std()
train_cp = (train_cp - cp_mean) / cp_std
val_cp = (val_cp - cp_mean) / cp_std
test_cp = (test_cp - cp_mean) / cp_std

conds_mean, conds_std = train_conds.mean(axis=0), train_conds.std(axis=0)
print(f"Condition means: {conds_mean}, stds: {conds_std}")
train_conds = (train_conds - conds_mean) / conds_std
val_conds = (val_conds - conds_mean) / conds_std
test_conds = (test_conds - conds_mean) / conds_std

coords = torch.tensor(mesh_coords, dtype=torch.float32)          # (N, 2), physical units
coords_mean, coords_std = coords.mean(dim=0), coords.std(dim=0)
node_coords = ((coords - coords_mean) / coords_std).T.contiguous().to(device)  # (2, N) node features
num_points = coords.shape[0]

# No padding needed (unlike the DiT): the graph works on the raw number of surface points.
train_dataset = TensorDataset(torch.tensor(train_cp).float(), torch.tensor(train_conds).float())
val_dataset = TensorDataset(torch.tensor(val_cp).float(), torch.tensor(val_conds).float())
test_dataset = TensorDataset(torch.tensor(test_cp).float(), torch.tensor(test_conds).float())

# Graph connectivity: kNN on the physical coordinates. All samples share the same mesh.
edge_index = knn_graph(coords, k=KNN_K, batch=None, loop=False).to(device)
print(f"Graph: {num_points} nodes, {edge_index.shape[1]} edges (k={KNN_K})")

coefficients = {
    "cp_mean": float(cp_mean),
    "cp_std": float(cp_std),
    "conds_mean": conds_mean.tolist(),
    "conds_std": conds_std.tolist(),
    "coords_mean": coords_mean.tolist(),
    "coords_std": coords_std.tolist(),
    "num_points": int(num_points),
    "knn_k": KNN_K,
    "h5_path": H5_PATH,
}
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(RESULTS_DIR / "coefficients.json", "w") as f:
    json.dump(coefficients, f, indent=2)
shutil.copy(__file__, RESULTS_DIR / os.path.basename(__file__))


# ----------------------------- graph helpers -----------------------------
@lru_cache(maxsize=None)
def get_batched_edge_index(batch_size):
    """edge_index for `batch_size` disjoint copies of the mesh graph (node ids offset by N per sample)."""
    num_edges = edge_index.size(1)
    offsets = torch.arange(batch_size, device=device).repeat_interleave(num_edges) * num_points
    return edge_index.repeat(1, batch_size) + offsets


def build_inputs(conds):
    """(B, 2) normalized conditions -> (B, 4, N) node features [x, z, alpha, mach]."""
    b = conds.shape[0]
    return torch.cat(
        [node_coords.unsqueeze(0).expand(b, -1, -1), conds.unsqueeze(-1).expand(-1, -1, num_points)],
        dim=1,
    )


@torch.no_grad()
def predict(model, dataset, batch_size=EVAL_BATCH_SIZE):
    """Returns normalized predictions with shape (S, 1, N) on CPU."""
    model.eval()
    preds = []
    for _, conds in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        conds = conds.to(device)
        out = model(build_inputs(conds), get_batched_edge_index(conds.shape[0]))
        preds.append(out.cpu())
    model.train()
    return torch.cat(preds, dim=0)


def cycle(dl):
    while True:
        for data in dl:
            yield data


# ----------------------------- model -----------------------------
model = ConditionedGraphUNet(
    dim=128,
    in_channels=4,  # x, z, alpha, mach
    out_channels=1,  # cp
    # cond_dim=None -> conditions are given as node features, not embeddings
    cond_drop_prob=0.0,
    attention_layers=[1, 2, 3],
    dim_mults=(1, 2, 2, 4),
    pool_ratios=0.5,
    sum_res=True,
    act=torch.nn.GELU(),
    attn_heads=16,
    attn_dim_head=64,
)
model.to(device)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print("Number trainable of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAINING_ITERS, eta_min=ETA_MIN)


# ----------------------------- training -----------------------------
if __name__ == "__main__":
    dl = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True))

    step = 0
    all_losses = []
    val_losses = []  # (step, normalized val MSE)
    best_val_loss = float("inf")
    model.train()
    with tqdm(total=TRAINING_ITERS) as pbar:
        while step < TRAINING_ITERS:
            targets, conds = next(dl)
            targets, conds = targets.to(device), conds.to(device)

            preds = model(build_inputs(conds), get_batched_edge_index(conds.shape[0]))  # (B, 1, N)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            step += 1
            all_losses.append(loss.item())
            pbar.set_description(f'loss: {loss.item():.5f}')
            pbar.update(1)

            if step % EVAL_EVERY == 0 or step == TRAINING_ITERS:
                val_preds = predict(model, val_dataset)
                val_loss = F.mse_loss(val_preds, val_dataset.tensors[0]).item()
                val_losses.append((step, val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), RESULTS_DIR / "gunet_state_dict_best.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss_history": all_losses,
                    "val_losses": val_losses,
                }, RESULTS_DIR / "gunet_ckp_last.pt")
                tqdm.write(f"step {step}: val MSE (normalized) {val_loss:.6f} | best {best_val_loss:.6f}")

    # loss curves
    plt.figure()
    plt.plot(all_losses, label='Train loss', alpha=0.5)
    window_size = 100
    if len(all_losses) >= window_size:
        moving_avg = np.convolve(all_losses, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(all_losses)), moving_avg, label=f'Moving Avg ({window_size})')
    if val_losses:
        plt.plot(*zip(*val_losses), 'o-', label='Validation loss')
    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE, normalized (log scale)')
    plt.title('Training Loss Evolution')
    plt.legend()
    plt.savefig(RESULTS_DIR / "loss_evolution.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # ----------------------------- test evaluation (best val checkpoint) -----------------------------
    model.load_state_dict(torch.load(RESULTS_DIR / "gunet_state_dict_best.pt", map_location=device))
    test_preds = predict(model, test_dataset)

    predictions_np = (test_preds.squeeze(1).numpy() * cp_std) + cp_mean          # (S_test, N)
    targets_np = (test_dataset.tensors[0].squeeze(1).numpy() * cp_std) + cp_mean  # (S_test, N)
    np.savez_compressed(
        RESULTS_DIR / "test_predictions.npz",
        predictions=predictions_np,
        targets=targets_np,
        conditions=all_conds[test_indices],  # (alpha, mach), not normalized
    )
    torch.save(torch.tensor(test_preds.squeeze(1)), RESULTS_DIR / "test_predictions_ema.pt")

    evaluator = RegressionEvaluator()
    metrics = evaluator(targets_np, predictions_np)
    evaluator.print_metrics()

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        if np.isscalar(obj) or isinstance(obj, np.generic):
            return float(obj)
        try:
            return to_serializable(obj.tolist())
        except Exception:
            try:
                return float(obj)
            except Exception:
                return obj

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(to_serializable(metrics), f, indent=2)
    print(f"Results saved to {RESULTS_DIR}")
