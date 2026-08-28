import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.video_dit import DiT_models, DiT
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

data = np.load("data/modulus_datasets_cylinder-flow_vv1/dataset/rawData.npy", allow_pickle=True)

fields = torch.from_numpy(data["x"]).float()
fields = fields.permute(0, 1, 3, 2) # put the channel dimension after the spatial dimensions (they where in last dim)
print("fields shape", fields.shape)

conds = torch.from_numpy(data["para"]).float().unsqueeze(-1) # add a dummy dimension for the channel (parameter) dimension
print("conds shape", conds.shape)

train_percent = 0.8
ntrain = int(train_percent*fields.shape[0])
ntest = fields.shape[0] - ntrain
fields_train = fields[:ntrain]
fields_test = fields[ntrain:]
conds_train = conds[:ntrain]
conds_test = conds[ntrain:]

# standardize the data per field independently
fields_train_mean = fields_train.mean(dim=(0, 1, 3), keepdim=True)
fields_train_std = fields_train.std(dim=(0, 1, 3), keepdim=True)
# not needed for latent training
# fields_train = (fields_train - fields_train_mean) / fields_train_std
# fields_test = (fields_test - fields_train_mean) / fields_train_std

conds_mean = conds_train.mean(dim=0, keepdim=True)
conds_std = conds_train.std(dim=0, keepdim=True)
conds_train = (conds_train - conds_mean) / conds_std
conds_test = (conds_test - conds_mean) / conds_std
print("conds_train shape", conds_train.shape)
print("conds_test shape", conds_test.shape)

print("fields_train mean", fields_train.mean(dim=(0, 1, 3)))
print("fields_train std", fields_train.std(dim=(0, 1, 3)))
print("conds_train mean", conds_train.mean(dim=0))
print("conds_train std", conds_train.std(dim=0))

# add padding
original_length = fields_train.shape[-1]
target_length = original_length + 1 
pad_length = target_length - original_length
# print("fields_train shape", fields_train.shape)
# print("fields_test shape", fields_test.shape)

# latents_train = F.pad(latents_train, (pad_length, 0))
# latents_test = F.pad(latents_test, (pad_length, 0))

vae_dir = "results/vae_cylinder/first_vae_good_attn_good_split/"
latents_train = np.load(vae_dir + 'latents_train.npy')
latents_test = np.load(vae_dir + 'latents_test.npy')
n_frames = fields_train.shape[1]

latents_train = latents_train.reshape(-1, n_frames, latents_train.shape[-2], latents_train.shape[-1])
latents_test = latents_test.reshape(-1, n_frames, latents_test.shape[-2], latents_test.shape[-1])

print("conds_train shape", conds_train.shape)
print("latents_train shape", latents_train.shape)
print("per channel latents_train mean/std", latents_train.mean(axis=(0, 2)), "\n", latents_train.std(axis=(0, 2)))

dataset_train = TensorDataset(
    torch.tensor(latents_train, dtype=torch.float32), torch.tensor(conds_train, dtype=torch.float32)
)
dataset_test = TensorDataset(
    torch.tensor(latents_test, dtype=torch.float32), torch.tensor(conds_test, dtype=torch.float32)
)  

model = DiT(
    depth=12,
    hidden_size=1280,
    patch_size=1,
    num_frames=fields_train.shape[1],
    num_heads=20,
    input_size=latents_train.shape[-1], # dataset grid size
    cond_dim=1, # number of parameters (Re)
    class_dropout_prob=0.2,
    in_channels=latents_train.shape[2],
    learn_sigma=False,
    use_swiglu=True,
    # use_rope=True,
    qk_norm=True, # when bf16 training
    attn_type="linear",  # window, linear, vanilla, physics
    mlp_ratio=3,
    factorize=True,
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=latents_train.shape[-1],
    cond_scale=2,
    num_sampling_steps=100,
    sampling_method="euler",
    # shifted_mu=1.0986
)

results_folder = 'results/cylinder_nvidia/ldm_depth_12_p1_hid1280_linear_factorize'

train_steps = 100000

trainer = Trainer1D(
    diffusion,
    dataset=dataset_train,
    dataset_test=dataset_test, # small_val_dataset is to avoid timeout when training on 2 GPUs
    train_batch_size=2,
    train_lr=1e-4,
    train_num_steps=train_steps,  # total training steps
    gradient_accumulate_every=4,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,     # turn on mixed precision
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

# trainer.load(20)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
trainer.train()

samples, seqs = trainer.eval_model(dataset_test, batch_size=16, use_autocast=True) # , cfg_interval_start=0.2

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
    # fold frame dimension into batch dimension for decoding
    samples = samples.reshape(-1, samples.shape[-2], samples.shape[-1])

    denormalized_latents = denormalise_latents(samples, vae_dir + "latents_stats.npz")
    cfg = AutoencoderKL1dConfig(
        in_channels=3,
        base_channels=192,
        num_heads=8,
        qk_norm=True,
        attention_resolutions=[0, 1, 2, 3],
        channel_multipliers=[1, 2, 4, 4],
        latent_channels=16,
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
    decoded_samples = decoded_samples[:, :, :original_length]  # unpad
    # add back the frame dimension
    decoded_samples = decoded_samples.reshape(-1, n_frames, decoded_samples.shape[-2], decoded_samples.shape[-1])
    print("decoded_samples shape", decoded_samples.shape)
    torch.save(decoded_samples, f"{results_folder}/test_predictions_ema.pt")
    denomralized_samples = (decoded_samples * fields_train_std) + fields_train_mean
    metrics = evaluator(denomralized_samples, fields_test[..., :original_length])
    evaluator.print_metrics()
