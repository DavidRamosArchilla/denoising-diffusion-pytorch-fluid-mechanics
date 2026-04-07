import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import evaluate_model
from denoising_diffusion_pytorch.dit import DiT_models, DiTCoordPE
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
import shutil
import os
import json


data_file = 'data/airfrans/processed_data.npz'
data = np.load(data_file)
pressures = data['pressures']
coords = data['coords']
conditions = data['conditions']

# split and process data
pressures = torch.from_numpy(pressures).float().unsqueeze(1)  # add channel dimension
coords = torch.from_numpy(coords).float()[:, 1, :]  # only care about y coordinates
conditions = torch.from_numpy(conditions).float()
n_train = int(0.8 * len(pressures))

pressures_train, pressures_val = pressures[:n_train], pressures[n_train:]
coords_train, coords_val = coords[:n_train], coords[n_train:]
conditions_train, conditions_val = conditions[:n_train], conditions[n_train:]

pressure_mean, pressure_std = pressures_train.mean(), pressures_train.std()
pressures_train = (pressures_train - pressure_mean) / pressure_std
pressures_val = (pressures_val - pressure_mean) / pressure_std

coords_min, coords_max = coords_train.min(), coords_train.max()
coords_train = (coords_train - coords_min) / (coords_max - coords_min)
coords_val = (coords_val - coords_min) / (coords_max - coords_min)

conds_mean, conds_std = conditions_train.mean(), conditions_train.std()
conditions_train = (conditions_train - conds_mean) / conds_std
conditions_val = (conditions_val - conds_mean) / conds_std

final_conds_train = torch.cat([coords_train, conditions_train], dim=1)
final_conds_val = torch.cat([coords_val, conditions_val], dim=1)
print(pressures_train.shape, final_conds_train.shape, conditions_train.shape, coords_train.shape)

train_dataset = TensorDataset(pressures_train, final_conds_train)
val_dataset = TensorDataset(pressures_val, final_conds_val)

model = DiT_models['DiT-XS/1'](
    input_size=train_dataset.tensors[0].shape[2],
    cond_dim=train_dataset.tensors[1].shape[1],
    class_dropout_prob=0.2,
    in_channels=1,
    learn_sigma=False,
    # use_bias=False,
    use_swiglu=True,
    use_rope=True,
    # qk_norm=True,
    attn_type="vanilla", # window, linear, vanilla
    # window_size=107,
    # num_experts=8,
    # num_experts_per_tok=2
    mlp_ratio=2.5
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=train_dataset.tensors[0].shape[2],
    cond_scale=2,
    num_sampling_steps=400,
    sampling_method="euler",
    # shifted_mu=1.0986
)

results_folder = 'results/airfrans/dit_xs_1_linear_rope'
train_steps = 200000
trainer = Trainer1D(
    diffusion,
    dataset=train_dataset,
    dataset_test=val_dataset, # small_val_dataset is to avoid timeout when training on 2 GPUs
    train_batch_size=64,
    train_lr=2e-4,
    num_samples=9,
    train_num_steps=train_steps,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    # amp=True,     # turn on mixed precision
    # mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    # use_cpu=True,
    # use_muon=True,
    compile_model=True,
    split_batches=True
)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
# with open(os.path.join(results_folder, 'norm_coefficients.json'), 'w') as f:
#     json.dump(coefficients, f, indent=4)
trainer.train(do_profiling=False)

if trainer.accelerator.is_main_process:
    diffusion = trainer.accelerator.unwrap_model(diffusion, keep_torch_compile=True)
    trainer.ema.ema_model.eval()  # Ensure eval mode
    diffusion.eval()
    original_length = 1063
    test_data, test_parameters = val_dataset.tensors
    errors, samples = evaluate_model(
        trainer.ema.ema_model, # trainer.ema.ema_model, # diffusion
        test_parameters,
        test_data,
        32,
        cond_scale=2
    )
    samples = samples[:, :original_length]  # Remove padding if it was added
    print(f"Final errors:\n{errors}")
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")

    from cetaceo.evaluators import RegressionEvaluator
    # preds = preds * (cp_max - cp_min) + cp_min
    samples = (samples * pressure_std) + pressure_mean
    test_data = (test_data * pressure_std) + pressure_mean

    evaluator = RegressionEvaluator()
    metrics = evaluator(samples, test_data[:, :original_length])
    # metrics = evaluator(preds, cp_test) # cp_test
    evaluator.print_metrics()