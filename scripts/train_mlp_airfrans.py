import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.dit import DiT_models, DiTCoordPE, DiTMultiShape, DiT, FinalLayer1D, CoordEmbedder
from denoising_diffusion_pytorch.vit import ViT
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
from data_generation.airfrans_dataset import AirfoilDataset
import shutil
import os
import json
import numpy as np
from torch.utils.data import TensorDataset
import pyLOM.NN
import optuna


# =========================
# TRAIN DATA
# =========================
data_file_train = 'data/airfrans/dataset_train_ordered.pt'
data_train = torch.load(data_file_train, weights_only=False)

pressures_train = data_train["pressures"]
coords_train = data_train["coords"]
conditions_train = data_train["conditions"]

repeats = [c.shape[0] for c in coords_train]

conditions_train = np.array(conditions_train)
conditions_train = np.repeat(conditions_train, repeats, axis=0)
coords_train = np.vstack(coords_train)
pressures_train = np.concatenate(pressures_train).reshape(-1, 1)

x_train = np.hstack((coords_train, conditions_train))
print("x_train shape:", x_train.shape)

# -----------------------------------
# Standardize using TRAIN statistics
# -----------------------------------
x_mean = x_train.mean(axis=0, keepdims=True)
x_std = x_train.std(axis=0, keepdims=True)

x_train_std = (x_train - x_mean) / x_std
print("Standardized x_train shape:", x_train_std.shape)

p_mean = pressures_train.mean(axis=0, keepdims=True)
p_std = pressures_train.std(axis=0, keepdims=True)

pressures_train_std = (pressures_train - p_mean) / p_std

# =========================
# TEST DATA
# =========================
data_file_test = 'data/airfrans/dataset_test_ordered.pt'
data_test = torch.load(data_file_test, weights_only=False)

pressures_test = data_test["pressures"]
coords_test = data_test["coords"]
conditions_test = data_test["conditions"]

repeats_test = [c.shape[0] for c in coords_test]

conditions_test = np.array(conditions_test)
conditions_test = np.repeat(conditions_test, repeats_test, axis=0)
coords_test = np.vstack(coords_test)
pressures_test = np.concatenate(pressures_test).reshape(-1, 1)

x_test = np.hstack((coords_test, conditions_test))
print("x_test shape:", x_test.shape)

# -----------------------------------
# Standardize TEST using TRAIN stats
# -----------------------------------
x_test_std = (x_test - x_mean) / x_std
pressures_test_std = (pressures_test - p_mean) / p_std
print("Standardized x_test shape:", x_test_std.shape)
print("Standardized pressures_test shape:", pressures_test_std.shape)

# =========================
# Convert to torch tensors
# =========================
x_train_tensor = torch.tensor(x_train_std, dtype=torch.float32)
y_train_tensor = torch.tensor(pressures_train_std, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_std, dtype=torch.float32)
y_test_tensor = torch.tensor(pressures_test_std, dtype=torch.float32)

# =========================
# TensorDatasets
# =========================
SEED = 42
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_dataset_split, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(SEED))
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

N_WORKERS = 15
DEVICE = "cuda"
CASE_NAME = "mlp"
CASE_DIR = "./results/airfrans_good_split/mlp"
optim_params= {
    "optimizer_class": torch.optim.AdamW,
    "loss_fn": torch.nn.MSELoss(),
    "activation": torch.nn.functional.elu,
    "initialization": torch.nn.init.kaiming_normal_,
    "initialization_kwargs": {"nonlinearity": "relu"},
    "scheduler_class": torch.optim.lr_scheduler.StepLR,
    "scheduler_kwargs": {"step_size": 1, "gamma": (0.9, 0.999)},
    "dataloader_kwargs": {"num_workers": N_WORKERS, "prefetch_factor": 5, "persistent_workers": True},
    "lr": (1e-4, 1e-2),
    "batch_size": (128, 2048),
    "hidden_size": (64, 256),
    "n_layers": (2, 5),
    "p_dropout": (1e-5, 0.1),
    "print_rate_epoch": 10,
    "epochs": (20, 300),
    "device": DEVICE,
    "seed": SEED,
    "model_name": CASE_NAME,
    "save_logs_path":CASE_DIR,
}

training_params = {
    "optimizer_class": torch.optim.AdamW,
    "loss_fn": torch.nn.MSELoss(),
    "activation": torch.nn.functional.elu,
    "initialization": torch.nn.init.kaiming_normal_,
    "initialization_kwargs": {"nonlinearity": "relu"},
    "scheduler_class": torch.optim.lr_scheduler.StepLR,
    "scheduler_kwargs": {"step_size": 1, "gamma": 0.99},
    "dataloader_kwargs": {"num_workers": N_WORKERS, "prefetch_factor": 5, "persistent_workers": True},
    "print_rate_epoch": 1,
    "device": DEVICE,
    "seed": SEED,
    "model_name": CASE_NAME,
    "save_logs_path": CASE_DIR,
    "lr": 1e-3,
    "batch_size": 256,
    "hidden_size": 256,
    "n_layers": 4,
    "p_dropout": 0.0,
    "epochs": 300,
}

optimizer = pyLOM.NN.OptunaOptimizer(
    optimization_params=optim_params,
    n_trials=200,
    direction="minimize",
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=optim_params["epochs"][1], reduction_factor=3),
    sampler = optuna.samplers.QMCSampler(seed=SEED, warn_independent_sampling=False),
    save_dir=str(CASE_DIR + '/hyperparameters'),
    # storage="sqlite:///" + optuna_file,
    # study_name=study_name,
)


model = pyLOM.NN.MLP(
    hidden_size=256,
    n_layers=4,
    input_size=4,
    output_size=1,
    device="cuda"
)
pipeline = pyLOM.NN.Pipeline(
    train_dataset=train_dataset_split,
    test_dataset=test_dataset,
    valid_dataset=val_dataset,
    # model_class=pyLOM.NN.MLP,
    # optimizer=optimizer,
    training_params=training_params,
    model=model
)

results = pipeline.run()
model = pipeline.model
evaluation_params = {
    "batch_size": 1024,
    "dataloader_kwargs": {
        "num_workers": N_WORKERS,
        "prefetch_factor": 5,
        "persistent_workers": True
    }
}
evaluator = pyLOM.NN.RegressionEvaluator()
_, variables_test = pipeline.evaluate(set_to_use="test", evaluators=[evaluator], evaluation_params=evaluation_params)
evaluator.print_metrics()
