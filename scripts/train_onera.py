import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch.distributed.distributed_c10d import destroy_process_group
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit import DiT_models, DiT, DiTBlock, DiTCoordPE
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
from denoising_diffusion_pytorch.trainer_hybrid import TrainerHybrid, TrainerCP
import shutil
import os
import pandas as pd

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)


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
coords = X_train_tot[0:nwallp,:3]

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

# reduce the number of geometric points
# num_points_to_keep = 75000
# selected_indices = np.random.choice(nwallp, num_points_to_keep, replace=False)
# Y_train = Y_train[:, selected_indices, :]
# Y_test = Y_test[:, selected_indices, :]

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
pad_length = 260864
Y_train = F.pad(Y_train, (0, pad_length - nwallp))
Y_test = F.pad(Y_test, (0, pad_length - nwallp))

print("X train shape", X_train.shape)
print("X test shape", X_test.shape)
print("Y train shape", Y_train.shape)
print("Y test shape", Y_test.shape)
print("mean/std X train test ", X_train.mean(axis=0), X_test.mean(axis=0), X_train.std(axis=0), X_test.std(axis=0))
print("mean/std Y train test ", Y_train.mean(dim=(0, 2)), Y_test.mean(dim=(0, 2)), Y_train.std(dim=(0, 2)), Y_test.std(dim=(0, 2)))
dataset_train = TensorDataset(
    Y_train, torch.tensor(X_train, dtype=torch.float32)
)

dataset_test = TensorDataset(
    Y_test, torch.tensor(X_test, dtype=torch.float32)
)

# model = Unet1D(
#     dim=128,
#     dim_mults=(1, 2, 2, 4),  # , 8),
#     # flash_attn = False,
#     channels=dataset_train.tensors[0].shape[1],  
#     cond_dim=dataset_train.tensors[1].shape[1],
#     cond_drop_prob=0.2,
#     attn_dim_head=64,
#     attn_heads=8,
#     learn_sigma=False,
#     # self_condition=True,
#     # full_attn = False
# )
model = DiT(
    depth=8,
    hidden_size=256,
    patch_size=2,
    num_heads=8,
    input_size=Y_train.shape[2],
    cond_dim=dataset_train.tensors[1].shape[1],
    class_dropout_prob=0.15,
    in_channels=dataset_train.tensors[0].shape[1],
    learn_sigma=False,
    use_swiglu=True,
    attn_type="linear", # linear
    qk_norm=True, # to avoid stability issues with bf16
    mlp_ratio=2.5,
    # num_experts=8,
    # num_experts_per_tok=2
)
# model = DiTCoordPE(
#     coords=torch.tensor(coords, dtype=torch.float32),
#     depth=12,
#     hidden_size=256,
#     patch_size=8,
#     num_heads=8,
#     input_size=Y_train.shape[2],
#     cond_dim=dataset_train.tensors[1].shape[1],
#     class_dropout_prob=0.2,
#     in_channels=dataset_train.tensors[0].shape[1],
#     learn_sigma=False,
#     # use_bias=False,
#     use_swiglu=True,
#     use_rope=True,
#     qk_norm=True,
#     attn_type="linear",  # window, linear, vanilla
#     # window_size=107,
#     # num_experts=8,
#     # num_experts_per_tok=2
#     mlp_ratio=4,
# )

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print("Number of learneable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

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
    input_size=dataset_train.tensors[0].shape[2],
    cond_scale=2,
    num_sampling_steps=100,
    sampling_method="euler",
    # shifted_mu=1.0986
)

results_folder = 'results/onera/dit_XS_p2_gradients'

train_steps = 100000

trainer = Trainer1D(
    diffusion,
    dataset=dataset_train,
    dataset_test=dataset_test,
    train_batch_size=8,
    train_lr=2e-4,
    num_samples=9,
    train_num_steps=train_steps,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp = True,                       # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    # use_cpu=True,
    use_muon=True,
    compile_model=True, 
    split_batches=True
)

# trainer = TrainerCP(
#     diffusion,
#     dataset=dataset_train,
#     cp_degree=2,
#     # dataset_test=dataset_test,
#     train_batch_size=4,
#     train_lr=2e-4,
#     train_num_steps=train_steps+4,  # total training steps
#     gradient_accumulate_every=4,  # gradient accumulation steps
#     ema_decay=0.995,  # exponential moving average decay
#     mixed_precision_type='bf16',
#     results_folder=results_folder,  # folder to save results to
#     save_and_sample_every=15000,
#     eta_min_scheduler=1e-6,
#     max_grad_norm=1.0,
#     # use_cpu=True,
#     # use_muon=True,
#     # compile_model=True, 
# )
# trainer.load(15)
trainer.train()
# shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))

trainer.ema.ema_model.eval()
# samples = trainer.ema.ema_model.sample(dataset_test.tensors[1][:1].to(trainer.device), return_all_steps=True)
samples, seqs = trainer.eval_model(dataset_test, batch_size=16, use_autocast=True)
# torch.save(samples, f"{results_folder}/samples_to_animation.pt")

if trainer.accelerator.is_main_process:
    # actual_size = len(dataset_test)
    print(samples.shape)
    print("NaNs: ", torch.isnan(samples).sum())
    test_data, test_parameters = dataset_test.tensors

    samples = samples[:, :, :original_length]  # unpad
    torch.save(samples, f"{results_folder}/test_predictions_ema_cfg_2_100_steps.pt")

    from cetaceo.evaluators import RegressionEvaluator
    # preds = preds * (cp_max - cp_min) + cp_min
    samples = (samples * train_std) + train_mean
    test_data = (test_data[:, :, :original_length] * train_std) + train_mean

    evaluator = RegressionEvaluator()
    metrics = evaluator(samples, test_data)
    # metrics = evaluator(preds, cp_test) # cp_test
    evaluator.print_metrics()

destroy_process_group()