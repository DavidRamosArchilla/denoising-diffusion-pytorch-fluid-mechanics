import torch 
torch.set_float32_matmul_precision('high')
import pyLOM
from torch.utils.data import TensorDataset, random_split
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit import DiT_models, DiT
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
import shutil
import os
import numpy as np


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True 

# def get_split_indices(split_name, split_data):
#     all_conditions = split_data["All"]
#     print(len(all_conditions), all_conditions)
#     data = split_data[split_name]
#     # print(data)
#     indices = []
#     for i in range(data.shape[0]):
#         print(data[i])
#         print(np.where(all_conditions == data[i]))
#         idx = np.where(all_conditions == data[i])[0][0]
#         indices.append(idx)
#     return indices
def get_split_indices(split_name, split_data):
    all_conditions = split_data["All"]
    data = split_data[split_name]

    lookup = {tuple(row): i for i, row in enumerate(all_conditions)}
    return [lookup[tuple(row)] for row in data]

def sanity_check_splits(split_data):
    train_idx = set(get_split_indices("Train", split_data))
    val_idx   = set(get_split_indices("Validation", split_data))
    test_idx  = set(get_split_indices("Test", split_data))

    # Check pairwise disjointness
    assert train_idx.isdisjoint(val_idx),  "Train and Validation overlap!"
    assert train_idx.isdisjoint(test_idx), "Train and Test overlap!"
    assert val_idx.isdisjoint(test_idx),   "Validation and Test overlap!"

    # Optional: check coverage
    all_idx = train_idx | val_idx | test_idx
    assert len(all_idx) == len(split_data["All"]), \
        "Splits do not cover all samples exactly once"

    print("✅ Sanity check passed: splits are disjoint and complete.")

data_path = "/home/airbus/final_data_airbus/v2/database/clean.h5"
split_data = np.load("data/cetaceo/best_train-val-test_split.npy", allow_pickle=True).item()
# split_data = np.load("data/aeronef/best_train-val-test_split.npy", allow_pickle=True).item()
training_indices = get_split_indices("Train", split_data)
validation_indices = get_split_indices("Validation", split_data)
test_indices = get_split_indices("Test", split_data)
print(len(training_indices), len(set(training_indices)))
print(len(validation_indices), len(set(validation_indices)))
print(len(test_indices), len(set(test_indices)))
sanity_check_splits(split_data)

# original_dataset = pyLOM.Dataset.load("/home/d.ramos/cetaceo_plane_data/batch1.h5")
original_dataset = pyLOM.Dataset.load("/home/airbus/final_data_airbus/v2/database/clean.h5")
original_cp = original_dataset["CoefPressure"]  # shape (num_samples, num_points)
print("Original Cp shape:", original_cp.shape)
htp_cp = original_cp[original_dataset['Zone'][:, -1] == 2]
print("HTP Cp shape:", htp_cp.shape)
# print(np.hstack((original_dataset.xyz, original_dataset['Zone'])).shape)
print(original_dataset.varnames, original_dataset.fieldnames)
print(len(original_dataset.get_variable("aoa")))

# import code; code.interact(local=locals())

def load_dataset(path, norm_coefficients, FL=None, pad_to=None):
    data = pyLOM.Dataset.load(path)
    aoa = data.get_variable('aoa')
    mach = data.get_variable('M')
    # filter htp zone
    cp = data["CoefPressure"][data['Zone'][:, -1] == 2].T
    if "cp_min" not in norm_coefficients:
        norm_coefficients["cp_min"] = cp.min()
        norm_coefficients["cp_max"] = cp.max()
    cp = (cp - norm_coefficients["cp_min"]) / (norm_coefficients["cp_max"] - norm_coefficients["cp_min"])
    # if "cp_mean" not in norm_coefficients:
    #     norm_coefficients["cp_mean"] = cp.mean()
    #     norm_coefficients["cp_std"] = cp.std()
    # cp = (cp - norm_coefficients["cp_mean"]) / norm_coefficients["cp_std"]

    if "mach_mean" not in norm_coefficients:
        norm_coefficients["mach_mean"] = mach.mean()
        norm_coefficients["mach_std"] = mach.std()
    mach = (mach - norm_coefficients["mach_mean"]) / norm_coefficients["mach_std"]

    if "aoa_mean" not in norm_coefficients:
        norm_coefficients["aoa_mean"] = aoa.mean()
        norm_coefficients["aoa_std"] = aoa.std()
    aoa = (aoa - norm_coefficients["aoa_mean"]) / norm_coefficients["aoa_std"]
    if pad_to is not None and cp.shape[1] < pad_to:
        pad_width = pad_to - cp.shape[1]
        cp = np.pad(cp, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    if FL is not None:
        fl_mask = np.array(data.get_variable('FL')) == FL
        aoa = aoa[fl_mask]
        mach = mach[fl_mask]
        cp = cp[fl_mask]

    cp_t = torch.from_numpy(cp).float().unsqueeze(1)  # add channel dimension
    # cp_t = cp_t[..., :-1]
    aoa_t = torch.from_numpy(aoa).float()
    mach_t = torch.from_numpy(mach).float()
    conditions = torch.stack([aoa_t, mach_t], dim=1)
    print(cp.min(), cp.max())
    print(aoa.mean(), aoa.std())
    print(mach.mean(), mach.std())
    print(cp_t.shape, conditions.shape)

    return TensorDataset(cp_t, conditions)

norm_coefficients = {}
# 217088 is the closest multiple of 256 (patch size) greater than 217003 (mesh points)
pad_length = 217088
clean_dataset = load_dataset(data_path, norm_coefficients, pad_to=pad_length, FL=310)
batch1_dataset = load_dataset("/home/d.ramos/cetaceo_plane_data/batch1.h5", norm_coefficients, pad_to=pad_length)
dataset = torch.utils.data.ConcatDataset([clean_dataset, batch1_dataset])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

data_ex = dataset_train[0]
print(len(dataset_train), len(dataset_test))
print(data_ex[0].shape, data_ex[1].shape)
model = DiT(
    depth=12,
    hidden_size=384,
    patch_size=16,
    num_heads=6,
    input_size=pad_length,
    cond_dim=2,
    class_dropout_prob=0.5,
    in_channels=1,
    learn_sigma=False,
    use_swiglu=True,
)

# diffusion = GaussianDiffusion1D(
#     model,
#     seq_length=dataset.tensors[0].shape[2],
#     objective="pred_noise",  # 'pred_noise' or 'pred_x0'
#     beta_schedule="cosine",
#     sampling_timesteps=1000,
#     timesteps=1000,  # number of steps
#     # use_cfg_plus_plus=True,
#     min_snr_loss_weight=True,
#     min_snr_gamma=5
# )

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=pad_length,
    cond_scale=6,
    num_sampling_steps=500,
    sampling_method="euler",
)

results_folder = 'results/cetaceo_cp/FM_dit_S_64_FL310'

train_steps = 300000

trainer = Trainer1D(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset_train,
    dataset_test=dataset_test,
    train_batch_size=8,
    train_lr=1e-4,
    num_samples=9,
    train_num_steps=train_steps+4,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp = True,                       # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    # use_cpu=True,
    # use_muon=True,
    compile_model=True
)

# trainer.load(10)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
trainer.train()

test_data, test_parameters = dataset_test[:]
errors, samples = evaluate_model(
    trainer.ema.ema_model, # trainer.ema.ema_model, # diffusion
    test_parameters,
    test_data,
    64,
    cond_scale=6
)
print(f"Final errors:\n{errors}")
torch.save(samples, f"{results_folder}/test_predictions_ema.pt")