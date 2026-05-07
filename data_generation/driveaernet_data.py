import os
import torch
import numpy as np
import pyvista as pv
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

# Must be at module level, not inside the class
def _load_mesh(args):
    idx, path = args
    mesh = pv.read(f"{path}.vtk")
    return idx, mesh.points.astype(np.float32), mesh.point_data['p'].astype(np.float32)


class MeshDataset(Dataset):
    def __init__(self, file_paths: list[str], stats: dict | None = None, num_workers: int = 16):
        coords_list    = [None] * len(file_paths)
        pressures_list = [None] * len(file_paths)

        print(f"Loading {len(file_paths)} meshes with {num_workers} processes...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_load_mesh, (i, p)): i
                   for i, p in enumerate(file_paths)}
            for future in tqdm(as_completed(futures), total=len(file_paths)):
                idx, xyz, p = future.result()
                coords_list[idx]    = xyz
                pressures_list[idx] = p

        # ... rest of __init__ unchanged
        # ------------------------------------------------------------------ #
        # Statistics (train set computes them, val/test reuse them)          #
        # ------------------------------------------------------------------ #
        if stats is None:
            all_xyz = np.concatenate(coords_list, axis=0)   # (sum_N, 3)
            all_p   = np.concatenate(pressures_list, axis=0)

            self.stats = {
                'xyz_mean': all_xyz.mean(0).astype(np.float32),   # (3,)
                'xyz_std':  all_xyz.std(0).astype(np.float32),
                'p_mean':   all_p.mean(0).astype(np.float32),     # (3,)
                'p_std':    all_p.std(0).astype(np.float32),
            }
        else:
            self.stats = stats

        s = self.stats
        print("Dataset stats:")
        print(f"  XYZ mean: {s['xyz_mean']}, std: {s['xyz_std']}")
        print(f"  P   mean: {s['p_mean']}, std: {s['p_std']}")
        # ------------------------------------------------------------------ #
        # Standardize and store as tensors (NOT padded — done in collate_fn) #
        # ------------------------------------------------------------------ #
        self.coords    = [
            torch.from_numpy((xyz - s['xyz_mean']) / s['xyz_std'])
            for xyz in coords_list
        ]
        self.pressures = [
            torch.from_numpy((p - s['p_mean']) / s['p_std'])
            for p in pressures_list
        ]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        """Returns (pressure, coords) — both (N_i, 3), unpadded."""
        return self.pressures[idx], self.coords[idx]

def mesh_collate_fn(batch, pad_multiple: int = 1):
    pressures, coords = zip(*batch)

    max_len    = max(p.shape[0] for p in pressures)
    padded_len = int(np.ceil(max_len / pad_multiple) * pad_multiple)

    def pad_to(tensors, length):
        B = len(tensors)
        extra = tensors[0].shape[1:]            # () if scalar field, (3,) if vector
        out = torch.zeros(B, length, *extra)
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t
        return out

    p_pad = pad_to(pressures, padded_len)       # (B, L) or (B, L, 3)
    c_pad = pad_to(coords,    padded_len)       # (B, L, 3)
    # add channel dimension for pressures
    p_pad = p_pad.unsqueeze(1)                      # (B, 1, L) or (B, 3, L)
    c_pad = c_pad.permute(0, 2, 1)                # (B, 3, L)
    lengths = torch.tensor([p.shape[0] for p in pressures])
    mask = torch.arange(padded_len).unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)

    return p_pad, c_pad, mask


def build_datasets(
    data_root: str,
    split_dir: str,
    cache_path: str | None = None,
) -> tuple[MeshDataset, MeshDataset, MeshDataset, dict]:
    """
    Builds train / validation / test datasets.

    Args:
        data_root  : root folder containing the ~10 subdirectories of .vtk files
        split_dir  : folder containing train.txt, validation.txt, test.txt
        cache_path : optional .pt file to cache loaded data (skips re-reading vtk)

    Returns:
        train_ds, val_ds, test_ds, stats
    """
    # ------------------------------------------------------------------ #
    # 1. Index every .vtk file: filename -> absolute path                 #
    # ------------------------------------------------------------------ #
    name_to_path = {}
    for folder in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder)
        print(f"Indexing folder: {folder_path}")
        if not os.path.isdir(folder_path):
            print(f"Warning: skipping non-directory {folder_path}")
            continue
        for fname in os.listdir(folder_path):
            if fname.endswith('.vtk'):
                if fname in name_to_path:
                    print(f"Warning: duplicate filename '{fname}' — keeping first occurrence")
                else:
                    fname = fname.strip().split('.')[0]  # remove extension
                    name_to_path[fname] = os.path.join(folder_path, fname)

    # ------------------------------------------------------------------ #
    # 2. Resolve split file names to paths                               #
    # ------------------------------------------------------------------ #
    def resolve_split(txt_file: str) -> list[str]:
        with open(txt_file) as f:
            names = [l.strip() for l in f if l.strip()]
        paths, missing = [], []
        for name in names:
            (paths if name in name_to_path else missing).append(
                name_to_path.get(name, name)
            )
        if missing:
            print(f"  [{os.path.basename(txt_file)}] {len(missing)} files not found: "
                  f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
        return paths

    train_paths = resolve_split(os.path.join(split_dir, 'train.txt'))
    val_paths   = resolve_split(os.path.join(split_dir, 'validation.txt'))
    test_paths  = resolve_split(os.path.join(split_dir, 'test.txt'))

    # ------------------------------------------------------------------ #
    # 3. Build datasets (train stats flow into val/test)                 #
    # ------------------------------------------------------------------ #
    if cache_path and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        train_ds = MeshDataset.__new__(MeshDataset)
        train_ds.coords    = cached['train_coords']
        train_ds.pressures = cached['train_pressures']
        train_ds.stats     = cached['stats']

        val_ds = MeshDataset.__new__(MeshDataset)
        val_ds.coords    = cached['val_coords']
        val_ds.pressures = cached['val_pressures']
        val_ds.stats     = cached['stats']

        test_ds = MeshDataset.__new__(MeshDataset)
        test_ds.coords    = cached['test_coords']
        test_ds.pressures = cached['test_pressures']
        test_ds.stats     = cached['stats']

    else:
        print("Building train dataset and computing statistics...")
        train_ds = MeshDataset(train_paths, stats=None)
        print("Building validation dataset...")
        val_ds   = MeshDataset(val_paths,   stats=train_ds.stats)
        print("Building test dataset...")
        test_ds  = MeshDataset(test_paths,  stats=train_ds.stats)

        if cache_path:
            torch.save({
                'train_coords':    train_ds.coords,
                'train_pressures': train_ds.pressures,
                'val_coords':      val_ds.coords,
                'val_pressures':   val_ds.pressures,
                'test_coords':     test_ds.coords,
                'test_pressures':  test_ds.pressures,
                'stats':           train_ds.stats,
            }, cache_path)
            print(f"Saved cache to {cache_path}")

    return train_ds, val_ds, test_ds, train_ds.stats

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from functools import partial
    train_ds, val_ds, test_ds, stats = build_datasets(
        data_root  = '/home/d.ramos/DrivAerNet/PressureVTK',
        split_dir  = '/home/d.ramos/DrivAerNet/splits',
        cache_path = '/home/d.ramos/denoising-diffusion-pytorch-fluid-mechanics/data/mesh_cache.pt',
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                            collate_fn=partial(mesh_collate_fn, pad_multiple=32))
    for pressure, coords, mask in train_loader:
        print("Batch shapes:", pressure.shape, coords.shape, mask.shape)
        break