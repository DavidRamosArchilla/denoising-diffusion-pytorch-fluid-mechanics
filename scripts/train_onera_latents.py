import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit import DiT_models, DiT
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
from denoising_diffusion_pytorch.autoencoder import AutoencoderKL1dConfig, AutoencoderKL1d, decode_latents, denormalise_latents
import shutil
import os
import pandas as pd
from tqdm import tqdm

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

nwallp = 260774  # number of points on the aircraft skin

data_dir = "/home/airbus/onera_data"
X_train_tot = np.load(data_dir + '/X_train.npy')
Y_train_tot = np.load(data_dir + '/Ytrain.npy')
X_test_tot = np.load(data_dir + '/X_test.npy')
Y_test = np.load(data_dir + '/Ytest.npy')

X_train = X_train_tot[0::nwallp,6:9]
X_test = X_test_tot[0::nwallp,6:9]

df_description = pd.read_csv(data_dir + '/describe_train_test_repartition_with_weights.csv', index_col=0)
df_test = df_description.loc[~df_description['Train']]
df_train = df_description.loc[df_description['Train']]
ncase = len(df_description)  # 468
ntest = len(df_test) # 156
ntrain = ncase-ntest  # 312

condition_mean = X_train.mean(axis=0, keepdims=True)
condition_std = X_train.std(axis=0, keepdims=True) 

X_train = (X_train - condition_mean) / condition_std
X_test = (X_test - condition_mean) / condition_std

Y_train_tot_conditions = np.array([Y_train_tot[nwallp*i:nwallp*(i+1),:] for i in range(ntrain)])
Y_test_tot_conditions = np.array([Y_test[nwallp*i:nwallp*(i+1),:] for i in range(ntest)])
Y_train = torch.tensor(Y_train_tot_conditions, dtype=torch.float32).permute(0, 2, 1)
Y_test = torch.tensor(Y_test_tot_conditions, dtype=torch.float32).permute(0, 2, 1)
train_mean = Y_train.mean(dim=(0, 2), keepdim=True)
train_std  = Y_train.std(dim=(0, 2), keepdim=True)

vae_dir = "results/vae_onera_full_attn/"
latents_train = np.load(vae_dir + 'latents_train.npy')
latents_test = np.load(vae_dir + 'latents_test.npy')

print("X_train shape", X_train.shape)
print("latents_train shape", latents_train.shape)
print("per channel latents_train mean/std", latents_train.mean(axis=(0, 2)), "\n", latents_train.std(axis=(0, 2)))

dataset_train = TensorDataset(
    torch.tensor(latents_train, dtype=torch.float32), torch.tensor(X_train, dtype=torch.float32)
)
dataset_test = TensorDataset(
    torch.tensor(latents_test, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
)  

# para mare nostrum
# depth=14,
# hidden_size=768,
# patch_size=1,
# num_heads=6,
model = DiT(
    depth=12,
    hidden_size=256,
    patch_size=1,
    num_heads=8,
    input_size=latents_train.shape[2],
    cond_dim=dataset_train.tensors[1].shape[1],
    class_dropout_prob=0.2,
    in_channels=dataset_train.tensors[0].shape[1],
    learn_sigma=False,
    use_swiglu=True,
    attn_type="linear", # linear vanilla triton_linear
    qk_norm=True, # to avoid stability issues with bf16
    mlp_ratio=4,
    use_rope=True
    # num_experts=4,
    # num_experts_per_tok=2
)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=dataset_train.tensors[0].shape[2],
    cond_scale=6,
    num_sampling_steps=400,
    sampling_method="euler",
    # shifted_mu=1.0986
)

results_folder = 'results/onera_ldm/dit_s_linear_trully_rope'

train_steps = 300000

trainer = Trainer1D(
    diffusion,
    dataset=dataset_train,
    # dataset_test=dataset_test,
    train_batch_size=16,
    train_lr=2e-4,
    num_samples=9,
    train_num_steps=train_steps,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp = True,                       # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=15000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    # use_cpu=True,
    # use_muon=True,
    compile_model=True, 
    split_batches=True
)

# trainer.load(20)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
trainer.train()

samples, seqs = trainer.eval_model(dataset_test, batch_size=32) # , cfg_interval_start=0.2

if trainer.accelerator.is_main_process:
    # actual_size = len(dataset_test)
    print(samples.shape)
    print("NaNs: ", torch.isnan(samples).sum())
    test_data, test_parameters = dataset_test.tensors

    torch.save(samples, f"{results_folder}/latent_predictions.pt")

    from cetaceo.evaluators import RegressionEvaluator

    evaluator = RegressionEvaluator()
    metrics = evaluator(samples, test_data)
    # metrics = evaluator(preds, cp_test) # cp_test
    print("Metrics on the latents: ")
    evaluator.print_metrics()

    denormalized_latents = denormalise_latents(samples, vae_dir + "latents_stats.npz")
    cfg = AutoencoderKL1dConfig(
        in_channels=4,
        base_channels=192,
        num_heads=8,
        qk_norm=True,
        attention_resolutions=[0, 1, 2, 3],
        channel_multipliers=[1, 2, 4, 4],
        latent_channels=8,
    )
    vae = AutoencoderKL1d(cfg)
    vae = torch.compile(vae)
    vae.load_state_dict(torch.load(vae_dir + "vae_best.pt", map_location="cpu")["model"])
    vae = vae.to(trainer.accelerator.device)
    decoded_samples = decode_latents(
        vae=vae, 
        z=denormalized_latents,
        batch_size=8,
        device=trainer.accelerator.device
    )
    decoded_samples = decoded_samples[:, :, :nwallp]  # unpad
    torch.save(decoded_samples, f"{results_folder}/test_predictions_ema.pt")
    denomralized_samples = (decoded_samples * train_std) + train_mean
    metrics = evaluator(denomralized_samples, Y_test)
    evaluator.print_metrics()
