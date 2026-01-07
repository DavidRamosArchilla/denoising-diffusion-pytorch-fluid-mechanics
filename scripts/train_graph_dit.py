import numpy as np
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit_graph import GraphDiT
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching

import shutil
import os
import matplotlib.pyplot as plt


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

def get_split_indices(split_name, split_data):
    all_conditions = split_data["All"]
    data = split_data[split_name]
    indices = []
    for i in range(data.shape[0]):
        idx = np.where(all_conditions == data[i])[0][0]
        indices.append(idx)
    return indices

data = np.load("data/aeronef/db_random.npy", allow_pickle=True).item()
field_name_to_predict = "Pressure" # "Pressure"
# cp = data["Cp"]
# train_size = 0.8
# training_indices = np.random.choice(cp.shape[0], int(cp.shape[0] * train_size), replace=False)
# test_indices = np.setdiff1d(np.arange(cp.shape[0]), training_indices)
split_data = np.load("data/aeronef/best_train-val-test_split.npy", allow_pickle=True).item()
training_indices = get_split_indices("Train", split_data)
val_indices = get_split_indices("Validation", split_data)
test_indices = get_split_indices("Test", split_data)

def load_dataset(indices, norm_coefficients, data):
    aoa = data["Alpha"][indices]
    vinf = data["Vinf"][indices]
    cp = data[field_name_to_predict][indices]
    if "cp_min" not in norm_coefficients:
        norm_coefficients["cp_min"] = cp.min()
        norm_coefficients["cp_max"] = cp.max()
    cp = (cp - norm_coefficients["cp_min"]) / (norm_coefficients["cp_max"] - norm_coefficients["cp_min"])

    if "vinf_mean" not in norm_coefficients:
        norm_coefficients["vinf_mean"] = vinf.mean()
        norm_coefficients["vinf_std"] = vinf.std()
    vinf = (vinf - norm_coefficients["vinf_mean"]) / norm_coefficients["vinf_std"]

    if "aoa_mean" not in norm_coefficients:
        norm_coefficients["aoa_mean"] = aoa.mean()
        norm_coefficients["aoa_std"] = aoa.std()
    aoa = (aoa - norm_coefficients["aoa_mean"]) / norm_coefficients["aoa_std"]

    # pad/truncate to length 27500
    target_len = 27499
    if field_name_to_predict == "Pressure" and cp.shape[1] < target_len:
        pad_width = target_len - cp.shape[1]
        cp = np.pad(cp, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    cp_t = torch.tensor(cp).float().unsqueeze(1)  # add channel dimension
    # cp_t = cp_t[..., :-1]
    aoa_t = torch.tensor(aoa).float()
    mach_t = torch.tensor(vinf).float()
    conditions = torch.stack([aoa_t, mach_t], dim=1)
    print(cp.min(), cp.max())
    print(aoa.mean(), aoa.std())
    print(vinf.mean(), vinf.std())
    print(cp_t.shape, conditions.shape)
    mesh_coords = torch.stack([torch.tensor(data["Xcoordinate"][0]), torch.tensor(data["Ycoordinate"][0])], dim=1)

    return TensorDataset(cp_t, conditions), mesh_coords

coefficients = {}
dataset, mesh_coords = load_dataset(training_indices, coefficients, data)
val_dataset, _ = load_dataset(val_indices, coefficients, data)
test_dataset, _ = load_dataset(test_indices, coefficients, data)

print(mesh_coords.shape)

model = GraphDiT(
    mesh_pos=mesh_coords,
    cond_dim=2,
    class_dropout_prob=0.2,
    in_channels=1,
    learn_sigma=False,
    hidden_size=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4.0,
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=dataset.tensors[0].shape[2],
    objective="pred_noise",  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=1000,
    timesteps=1000,  # number of steps
    # use_cfg_plus_plus=True,
    min_snr_loss_weight=True,
    min_snr_gamma=5
)
sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

# diffusion = FlowMatching(
#     sampler,
#     model,
#     input_size=27499,
#     cond_scale=6,
#     num_sampling_steps=500,
#     sampling_method="euler",
# )

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 1e9} GB")

results_folder = 'results/aeronef_pressure_1d/graph_dit_FM'

train_steps = 150000

trainer = Trainer1D(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset,
    dataset_test=val_dataset,
    train_batch_size=16,
    train_lr=1e-4,
    num_samples=9,
    train_num_steps=train_steps+4,  # total training steps
    gradient_accumulate_every=4,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=15000,
    use_lr_scheduler=False,
    max_grad_norm=1.0,
    # use_cpu=True,
)

# trainer.load(10)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
trainer.train()
# trainer.load(9)

# torch.cuda.empty_cache()  # Clear GPU memory
trainer.ema.ema_model.eval()  # Ensure eval mode
diffusion = trainer.accelerator.unwrap_model(diffusion)
diffusion.eval()

test_data, test_parameters = test_dataset.tensors
errors, samples = evaluate_model(
    trainer.ema.ema_model, # trainer.ema.ema_model, # diffusion
    test_parameters,
    test_data,
    32,
    cond_scale=6
)
print(f"Final errors:\n{errors}")
torch.save(samples, f"{results_folder}/test_predictions_ema.pt")