"""
Train a deterministic Graph U-Net (ConditionedGraphUNet) on the TIFON dataset to
regress the surface pressure coefficient from the flow conditions (alpha, mach).

- Data loading, train/val/test splits and normalization follow scripts/train_tifon.py.
- The model / training loop follow scripts/train_gunet_onera.py: every node of the
  graph is a surface point with input features [x, z, alpha, mach] and the network
  predicts cp at that node (MSE loss, no diffusion).

Options (defaults reproduce the plain run: TopK pooling, features [x, z, alpha, mach]):
  --pooling stride   keep every other point along the surface instead of TopK pooling
  --normals          add the outward surface normal (nx, nz) as node features
  --arc-freqs K      add sin/cos(2^k * theta), k = 0..K-1, of the arc-length position
                     theta = 2*pi*s/L (s measured from the leading edge, L = perimeter)
  --out-head linear  per-point linear output layer instead of a GCNConv, so the output is
                     not averaged with the neighbours (sharp jumps, e.g. at the leading edge)
e.g. python scripts/train_gunet_tifon.py --pooling stride --normals --arc-freqs 8 --out-head linear
"""
import pyLOM

import os
import json
import math
import shutil
import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ema_pytorch import EMA

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
RESULTS_ROOT = Path("results/tifon_final")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--pooling", choices=["topk", "stride"], default="topk",
                    help="topk: learned TopK pooling; stride: keep every other point along the surface")
parser.add_argument("--normals", action="store_true", help="add the outward surface normal (nx, nz) as node features")
parser.add_argument("--arc-freqs", type=int, default=0,
                    help="number of sin/cos frequency pairs of the arc-length position (0 = none)")
parser.add_argument("--surface-hops", type=int, default=1,
                    help="connect each point to this many neighbours on each side along the surface")
parser.add_argument("--out-head", choices=["gcn", "linear"], default="gcn",
                    help="output layer: gcn (GCNConv, averages each point with its neighbours) or linear (per point)")
parser.add_argument("--results-dir", type=Path, default=None,
                    help="default: results/tifon_final/gunet_attn_h16_<options>_<training recipe>")
args = parser.parse_args()

POOLING = args.pooling
USE_NORMALS = args.normals
ARC_FREQS = args.arc_freqs
SURFACE_HOPS = args.surface_hops
OUT_HEAD = args.out_head

# Training recipe. Previous runs (lr 1e-4, 30-40k steps, no EMA) were still improving when the
# cosine schedule reached ETA_MIN, and the validation MSE jumped by up to 1.5x between checks.
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
TRAINING_ITERS = 60000    # was 40000
LR = 3e-4                 # peak learning rate (was 1e-4)
WARMUP_ITERS = 1000       # linear warmup to LR, then cosine decay to ETA_MIN
ETA_MIN = 1e-6
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
EMA_DECAY = 0.995         # EMA of the weights, as in train_tifon.py; the EMA model is the one
EMA_UPDATE_EVERY = 10     # validated, saved as best checkpoint and evaluated on test
EVAL_EVERY = 5000         # validation + checkpoint frequency (in steps), was 5000

# Model size. MODEL_DIM is the width of the first level; level i has MODEL_DIM * DIM_MULTS[i] channels.
# The attention layers have their own width (ATTN_HEADS * ATTN_DIM_HEAD) and dominate the run time.
MODEL_DIM = 256           # was 128 (9.7M params -> 20.9M); same run time, the attention size is unchanged
DIM_MULTS = (1, 2, 2, 4)
ATTENTION_LAYERS = [1, 2, 3]
ATTN_HEADS = 16
ATTN_DIM_HEAD = 64

run_name = f"gunet_attn_h{ATTN_HEADS}_" + POOLING
if SURFACE_HOPS != 1:
    run_name += f"_hops{SURFACE_HOPS}"
if USE_NORMALS:
    run_name += "_normals"
if ARC_FREQS > 0:
    run_name += f"_arc{ARC_FREQS}"
if OUT_HEAD != "gcn":
    run_name += f"_{OUT_HEAD}head"
if MODEL_DIM != 128:
    run_name += f"_dim{MODEL_DIM}"
run_name += f"_lr{LR:.0e}_{TRAINING_ITERS // 1000}k_ema"   # recipe, so new runs don't overwrite old ones
RESULTS_DIR = args.results_dir or RESULTS_ROOT / run_name

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
num_points = coords.shape[0]

# The graph and the geometric features below assume the points are ordered along the closed
# surface, starting at the leading edge. Fail loudly if they are not.
spacing = (coords - coords.roll(-1, dims=0)).norm(dim=1)   # includes the wrap N-1 -> 0
assert spacing.max() < 5 * spacing.median(), (
    f"Points do not look ordered along the surface: max spacing {spacing.max():.2e} "
    f"vs median {spacing.median():.2e} (at index {spacing.argmax().item()})"
)


# Geometric node features.
def surface_normals(xz):
    """Outward unit normals (N, 2) from central differences along the closed curve."""
    tangent = xz.roll(-1, dims=0) - xz.roll(1, dims=0)
    tangent = tangent / tangent.norm(dim=1, keepdim=True)
    normal = torch.stack([tangent[:, 1], -tangent[:, 0]], dim=1)   # outward for a counter-clockwise loop
    nxt = xz.roll(-1, dims=0)
    signed_area = 0.5 * (xz[:, 0] * nxt[:, 1] - nxt[:, 0] * xz[:, 1]).sum()
    return normal if signed_area > 0 else -normal


def arc_length_features(xz, num_freqs):
    """sin/cos(2^k * theta), k < num_freqs, with theta = 2*pi*s/L (s from point 0, L = perimeter).

    theta is periodic, so the features are continuous where the loop closes (the leading edge);
    the higher frequencies resolve small distances along the surface, e.g. around the leading edge.
    """
    seg = (xz.roll(-1, dims=0) - xz).norm(dim=1)                   # seg[i] = |p[i+1] - p[i]|, wrapping
    s = torch.cat([seg.new_zeros(1), seg[:-1].cumsum(0)])
    theta = 2 * math.pi * s / seg.sum()
    angles = theta[:, None] * (2.0 ** torch.arange(num_freqs))[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)           # (N, 2 * num_freqs)


feature_names = ["x", "z"]
features = [(coords - coords_mean) / coords_std]                    # standardized coordinates
if USE_NORMALS:
    features.append(surface_normals(coords))                        # unit vectors, used as is
    feature_names += ["nx", "nz"]
if ARC_FREQS > 0:
    features.append(arc_length_features(coords, ARC_FREQS))         # already in [-1, 1]
    feature_names += [f"sin{2**k}" for k in range(ARC_FREQS)] + [f"cos{2**k}" for k in range(ARC_FREQS)]
node_features = torch.cat(features, dim=1).T.contiguous().to(device)  # (F, N)
in_channels = node_features.shape[0] + 2                               # + alpha, mach
print(f"Node features ({in_channels}): {feature_names + ['alpha', 'mach']}")

# No padding needed (unlike the DiT): the graph works on the raw number of surface points.
train_dataset = TensorDataset(torch.tensor(train_cp).float(), torch.tensor(train_conds).float())
val_dataset = TensorDataset(torch.tensor(val_cp).float(), torch.tensor(val_conds).float())
test_dataset = TensorDataset(torch.tensor(test_cp).float(), torch.tensor(test_conds).float())

# Graph connectivity. The surface points are stored in order along the airfoil (closed loop
# starting and ending at the leading edge), so neighbours are taken by index instead of kNN:
# point i <-> i±1..±SURFACE_HOPS, wrapping N-1 <-> 0 at the leading edge. Upper and lower
# surfaces can then never be linked, however close they are. All samples share the same mesh.
def surface_ring_graph(n, hops):
    i = torch.arange(n)
    src, dst = [], []
    for h in range(1, hops + 1):
        j = (i + h) % n
        src += [i, j]
        dst += [j, i]
    return torch.stack([torch.cat(src), torch.cat(dst)])

edge_index = surface_ring_graph(num_points, SURFACE_HOPS).to(device)
print(f"Graph: {num_points} nodes, {edge_index.shape[1]} edges (ordered ring, ±{SURFACE_HOPS} neighbours)")

coefficients = {
    "cp_mean": float(cp_mean),
    "cp_std": float(cp_std),
    "conds_mean": conds_mean.tolist(),
    "conds_std": conds_std.tolist(),
    "coords_mean": coords_mean.tolist(),
    "coords_std": coords_std.tolist(),
    "num_points": int(num_points),
    "graph": "surface_ring",
    "surface_hops": SURFACE_HOPS,
    "pooling": POOLING,
    "normals": USE_NORMALS,
    "arc_freqs": ARC_FREQS,
    "out_head": OUT_HEAD,
    "model": {
        "dim": MODEL_DIM, "dim_mults": list(DIM_MULTS), "attention_layers": ATTENTION_LAYERS,
        "attn_heads": ATTN_HEADS, "attn_dim_head": ATTN_DIM_HEAD,
    },
    "node_features": feature_names + ["alpha", "mach"],
    "training": {
        "batch_size": BATCH_SIZE, "iters": TRAINING_ITERS, "lr": LR, "warmup_iters": WARMUP_ITERS,
        "eta_min": ETA_MIN, "weight_decay": WEIGHT_DECAY, "max_grad_norm": MAX_GRAD_NORM,
        "ema_decay": EMA_DECAY, "ema_update_every": EMA_UPDATE_EVERY,
    },
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
    """(B, 2) normalized conditions -> (B, F + 2, N) node features [geometry..., alpha, mach]."""
    b = conds.shape[0]
    return torch.cat(
        [node_features.unsqueeze(0).expand(b, -1, -1), conds.unsqueeze(-1).expand(-1, -1, num_points)],
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
    dim=MODEL_DIM,
    in_channels=in_channels,  # geometric features + alpha, mach
    out_channels=1,  # cp
    # cond_dim=None -> conditions are given as node features, not embeddings
    cond_drop_prob=0.0,
    attention_layers=ATTENTION_LAYERS,
    dim_mults=DIM_MULTS,
    pool_ratios=0.5,
    sum_res=True,
    act=torch.nn.GELU(),
    attn_heads=ATTN_HEADS,
    attn_dim_head=ATTN_DIM_HEAD,
    pool_type=POOLING,
    out_head=OUT_HEAD,
)
model.to(device)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print("Number trainable of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=WEIGHT_DECAY)


def lr_factor(step):
    """Linear warmup over WARMUP_ITERS steps, then cosine decay from LR to ETA_MIN at TRAINING_ITERS."""
    if step < WARMUP_ITERS:
        return (step + 1) / WARMUP_ITERS
    progress = min(1.0, (step - WARMUP_ITERS) / max(1, TRAINING_ITERS - WARMUP_ITERS))
    return ETA_MIN / LR + (1 - ETA_MIN / LR) * 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
ema = EMA(model, beta=EMA_DECAY, update_every=EMA_UPDATE_EVERY)
ema.to(device)


# ----------------------------- training -----------------------------
if __name__ == "__main__":
    dl = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True))

    step = 0
    all_losses = []
    val_losses = []      # (step, normalized val MSE) of the EMA model, used to pick the best checkpoint
    val_losses_raw = []  # (step, normalized val MSE) of the raw (online) model, for reference
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
            ema.update()

            step += 1
            all_losses.append(loss.item())
            pbar.set_description(f'loss: {loss.item():.5f} lr: {scheduler.get_last_lr()[0]:.2e}')
            pbar.update(1)

            if step % EVAL_EVERY == 0 or step == TRAINING_ITERS:
                val_targets = val_dataset.tensors[0]
                val_loss = F.mse_loss(predict(ema.ema_model, val_dataset), val_targets).item()
                val_loss_raw = F.mse_loss(predict(model, val_dataset), val_targets).item()
                val_losses.append((step, val_loss))
                val_losses_raw.append((step, val_loss_raw))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(ema.ema_model.state_dict(), RESULTS_DIR / "gunet_state_dict_best.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema.ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "loss_history": all_losses,
                    "val_losses": val_losses,
                    "val_losses_raw": val_losses_raw,
                }, RESULTS_DIR / "gunet_ckp_last.pt")
                tqdm.write(f"step {step}: val MSE (normalized) EMA {val_loss:.6f} | raw {val_loss_raw:.6f} "
                           f"| best EMA {best_val_loss:.6f}")

    # loss curves
    plt.figure()
    plt.plot(all_losses, label='Train loss', alpha=0.5)
    window_size = 100
    if len(all_losses) >= window_size:
        moving_avg = np.convolve(all_losses, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(all_losses)), moving_avg, label=f'Moving Avg ({window_size})')
    if val_losses:
        plt.plot(*zip(*val_losses_raw), 'o-', label='Validation loss (raw model)', alpha=0.6)
        plt.plot(*zip(*val_losses), 'o-', label='Validation loss (EMA model)')
    plt.yscale('log')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE, normalized (log scale)')
    plt.title('Training Loss Evolution')
    plt.legend()
    plt.savefig(RESULTS_DIR / "loss_evolution.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # ----------------------------- test evaluation (best val checkpoint, EMA weights) -----------------------------
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
