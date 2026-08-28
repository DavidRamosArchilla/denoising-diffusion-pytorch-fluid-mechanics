"""
Optuna hyperparameter search for the DiT + FlowMatching training pipeline.

Assumes the following are already defined/imported before this script runs
(exactly as in your original training script):
    - DiT, Sampler, create_transport, FlowMatching, Trainer1D
    - train_dataset, val_dataset
    - train_cp, cp_mean, cp_std, original_length

This script only adds the HPO layer on top of your existing pipeline; the
dataset / normalization constants are untouched.
"""

import os
import gc
import math
import json
from pathlib import Path
import shutil

import pyLOM.NN
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit import DiT_models, DiTCoordPE, DiT
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching

import torch
from torch.utils.data import TensorDataset, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt

import optuna
from optuna.trial import TrialState
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

# ----------------------------- CONFIG ---------------------------------
RESULTS_FOLDER = "results/tifon_new_split_ordered/optuna_full_training_batch1"
STUDY_NAME = "dit_flow_matching_hpo_full_training"
STORAGE_PATH = f"sqlite:///{RESULTS_FOLDER}/study.db"
N_TRIALS = 100

# Full training uses 100_000 steps. Running 50 full trainings at that length
# is almost certainly not feasible, so each HPO trial trains on a reduced
# step budget used only as a proxy signal to rank configurations. Increase
# this if you have the compute for it, or lower it further if trials are
# still too slow. Once you pick a winning config, retrain it separately with
# the full train_steps.
TRAIN_STEPS_PER_TRIAL = 100_000

os.makedirs(RESULTS_FOLDER, exist_ok=True)


def get_split_indices(split_name, split_data, all_conds, subsplits=[],atol=1e-4, rtol=1e-12):
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

    # Check pairwise disjointness
    assert train_idx.isdisjoint(val_idx),  "Train and Validation overlap!"
    assert train_idx.isdisjoint(test_idx), "Train and Test overlap!"
    assert val_idx.isdisjoint(test_idx),   "Validation and Test overlap!"

    # Optional: check coverage
    all_idx = train_idx | val_idx | test_idx
    assert len(all_idx) == len(split_data["all"]), \
        "Splits do not cover all samples exactly once"

    print("✅ Sanity check passed: splits are disjoint and complete.")

DATA_DIR  = Path("/home/airbus/CETACEO_cp_interp/DATA/TIFON")

# data = pyLOM.Dataset.load(DATA_DIR / "CADGroup_3_completo_stage_1.h5")
# data = pyLOM.Dataset.load("/home/airbus/CETACEO_cp_interp/CLUSTERING_TIFON/AIRBUS/database/CADGroup_3_completo_stage_1_reordered.h5")
data = pyLOM.Dataset.load("/home/airbus/CETACEO_cp_interp/TIFON/DATA/CADGroup_3_#280_stage_0_reordered.h5")
coords = data.xyz[:, [0, 2]]
cp = data["BoundaryValues_CoefPressure"].T
cp = cp[:, np.newaxis, :]
print("El nuevo dataset tiene estas claves, gracias por cambiarlas ", data._vardict.keys())
alpha, mach = torch.tensor(data.get_variable('aoa')).float(), torch.tensor(data.get_variable('mach')).float()
print(f"Loaded data: cp shape {cp.shape}, alpha shape {alpha.shape}, mach shape {mach.shape}")

# split_data = np.load(DATA_DIR / 'best_train-val-test_split.npy', allow_pickle=True).item()
# with open(DATA_DIR / "tifon_split_complete.json") as f:
with open(DATA_DIR / "/home/airbus/CETACEO_cp_interp/TIFON/splitting/tifon_split_complete.json") as f:
    split_data = json.load(f)
    print(split_data.keys())
    print(split_data["train"].keys())
    print(split_data["validation"].keys(), type(split_data["validation"]["batch0"]))

all_conds = torch.stack((alpha, mach), dim=1).numpy()
train_indices = get_split_indices("train", split_data, all_conds, ["batch0", "batch1"])
val_indices = get_split_indices("validation", split_data, all_conds, ["batch0", "batch1"])
test_indices = get_split_indices("test", split_data, all_conds)
sanity_check_splits(split_data, all_conds, ["batch0", "batch1", "batch2"])

train_cp, train_conds = cp[train_indices], all_conds[train_indices]
val_cp, val_conds = cp[val_indices], all_conds[val_indices]
test_cp, test_conds = cp[test_indices], all_conds[test_indices]

cp_mean, cp_std = train_cp.mean(), train_cp.std()
train_cp = (train_cp - cp_mean) / cp_std
val_cp = (val_cp - cp_mean) / cp_std
test_cp = (test_cp - cp_mean) / cp_std

original_length = cp.shape[2]
pad_length = 13872 # divisible by 16
train_cp = np.pad(train_cp, ((0, 0), (0, 0), (0, pad_length - train_cp.shape[2])), mode='constant')
val_cp = np.pad(val_cp, ((0, 0), (0, 0), (0, pad_length - val_cp.shape[2])), mode='constant')
test_cp = np.pad(test_cp, ((0, 0), (0, 0), (0, pad_length - test_cp.shape[2])), mode='constant')
print(f"After padding, cp shapes: train {train_cp.shape}, val {val_cp.shape}, test {test_cp.shape}")

conds_mean, conds_std = train_conds.mean(axis=0), train_conds.std(axis=0)
print(f"Condition means: {conds_mean}, stds: {conds_std}")
train_conds = (train_conds - conds_mean) / conds_std
val_conds = (val_conds - conds_mean) / conds_std
test_conds = (test_conds - conds_mean) / conds_std
print(f"After normalization, condition means: {train_conds.mean(axis=0)}, stds: {train_conds.std(axis=0)}")

train_dataset = TensorDataset(torch.tensor(train_cp).float(), torch.tensor(train_conds).float())
val_dataset = TensorDataset(torch.tensor(val_cp).float(), torch.tensor(val_conds).float())
test_dataset = TensorDataset(torch.tensor(test_cp).float(), torch.tensor(test_conds).float())

train_dataset_merged = ConcatDataset([train_dataset, val_dataset])


def objective(trial: optuna.Trial) -> float:
    # ------------------------- search space -------------------------
    depth = trial.suggest_int("depth", 4, 8)

    # Sample num_heads and head_dim separately and multiply them so that
    # hidden_size is divisible by num_heads by construction.
    head_dim = trial.suggest_categorical("head_dim", [32, 64])
    num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
    hidden_size = head_dim * num_heads
    trial.set_user_attr("hidden_size", hidden_size)  # log the derived value

    slice_num = trial.suggest_categorical("slice_num", [64, 128, 256])
    patch_size = trial.suggest_categorical("patch_size", [8, 16])
    class_dropout_prob = trial.suggest_float("class_dropout_prob", 0.1, 0.2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    # Kept narrow on purpose given the limited trial budget.
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    mlp_ratio = trial.suggest_float("mlp_ratio", 2, 4)

    model = sampler = diffusion = trainer = None

    try:
        model = DiT(
            depth=depth,
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_heads=num_heads,
            input_size=train_cp.shape[2],
            cond_dim=2,
            class_dropout_prob=class_dropout_prob,
            in_channels=1,
            learn_sigma=False,
            use_swiglu=True,
            attn_type="physics",
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
        )

        sampler = Sampler(transport=create_transport())

        diffusion = FlowMatching(
            sampler,
            model,
            input_size=train_cp.shape[2],
            cond_scale=2,
            num_sampling_steps=100,
            sampling_method="euler",
        )

        trial_folder = f"{RESULTS_FOLDER}/trial_{trial.number}"

        trainer = Trainer1D(
            diffusion,
            dataset=train_dataset,
            train_batch_size=batch_size,
            train_lr=learning_rate,
            train_num_steps=TRAIN_STEPS_PER_TRIAL,
            gradient_accumulate_every=1,
            ema_decay=0.995,
            results_folder=trial_folder,
            save_and_sample_every=TRAIN_STEPS_PER_TRIAL,  # only checkpoint at the end during HPO
            eta_min_scheduler=4e-6,
            max_grad_norm=1.0,
            use_muon=True,
            compile_model=True,
            split_batches=True,
        )

        trainer.train()

        samples, seqs = trainer.eval_model(val_dataset, batch_size=32, use_autocast=True)
        samples = samples[:, :original_length]
        samples = (samples * cp_std) + cp_mean

        test_data, test_parameters = val_dataset.tensors
        test_data = (test_data * cp_std) + cp_mean

        mse = ((samples - test_data[:, :original_length]) ** 2).mean().item()
        rmse = np.sqrt(mse)

        if not math.isfinite(rmse):
            raise optuna.exceptions.TrialPruned("Non-finite MSE (training likely diverged)")

        return rmse

    except torch.cuda.OutOfMemoryError:
        raise optuna.exceptions.TrialPruned("CUDA OOM for this configuration")

    finally:
        del model, sampler, diffusion, trainer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,  # resumes automatically if the DB already exists
    )

    n_completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    n_remaining = max(0, N_TRIALS - n_completed)
    print(
        f"[Optuna] {n_completed} trial(s) already complete for '{STUDY_NAME}', "
        f"running {n_remaining} more (target: {N_TRIALS})."
    )

    if n_remaining > 0:
        study.optimize(
            objective,
            n_trials=n_remaining,
            catch=(Exception,),  # a crashing trial fails that trial, not the whole study
        )

    print("\nBest trial:")
    print(f"  MSE: {study.best_trial.value:.6f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"    hidden_size (derived): {study.best_trial.user_attrs.get('hidden_size')}")

    # ----------------------------- plots -----------------------------
    plots_dir = f"{RESULTS_FOLDER}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    plot_optimization_history(study).write_html(f"{plots_dir}/optimization_history.html")
    plot_param_importances(study).write_html(f"{plots_dir}/param_importances.html")
    plot_parallel_coordinate(study).write_html(f"{plots_dir}/parallel_coordinate.html")

    print(f"\nPlots saved to {plots_dir}/")