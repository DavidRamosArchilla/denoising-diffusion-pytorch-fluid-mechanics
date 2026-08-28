import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.dit import DiT_models, DiTCoordPE, DiTMultiShape
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
from data_generation.airfrans_dataset import AirfoilDataset
import shutil
import os
import json


data_file_train = 'data/airfrans/dataset_train_ordered.pt' # ordered_dataset processed_data
data_train = torch.load(data_file_train, weights_only=False)
pressures_train, coords_train, conditions_train = data_train["pressures"], data_train["coords"], data_train["conditions"]
train_dataset = AirfoilDataset(pressures_train, coords_train, conditions_train)

coefficients = {
    'p_mean': train_dataset.p_mean,
    'p_std': train_dataset.p_std,
    'xy_mean': train_dataset.xy_mean,
    'xy_std': train_dataset.xy_std,
    'c_mean': train_dataset.c_mean,
    'c_std': train_dataset.c_std
}

data_file_test = 'data/airfrans/dataset_test_ordered.pt' # ordered_dataset processed_data
data_test = torch.load(data_file_test, weights_only=False)
pressures_test, coords_test, conditions_test = data_test["pressures"], data_test["coords"], data_test["conditions"]
test_dataset = AirfoilDataset(pressures_test, coords_test, conditions_test, coefficients=coefficients, max_len=train_dataset.max_len)

# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
data_sample = train_dataset[:]
print(f"Data sample: pressure shape {data_sample[0].shape}, condition shape {data_sample[1].shape}, coords shape {data_sample[2].shape}, mask shape {data_sample[3].shape}")

model = DiTMultiShape(
    depth=6,
    hidden_size=128,
    patch_size=1,
    num_heads=4,
    context_channels=data_sample[2].shape[1],
    input_size=data_sample[0].shape[2],
    cond_dim=data_sample[1].shape[1],
    class_dropout_prob=0.2,
    in_channels=1,
    learn_sigma=False,
    # use_bias=False,
    use_swiglu=True,
    use_rope=True,
    # qk_norm=True,
    attn_type="physics",  # window, linear, vanilla, physics
    slice_num=128,
    # window_size=107,
    # num_experts=8,
    # num_experts_per_tok=2
    mlp_ratio=4,
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=data_sample[0].shape[2],
    cond_scale=2,
    num_sampling_steps=400,
    sampling_method="euler",
    # shifted_mu=1.0986
)

results_folder = 'results/airfrans_good_split/dit_xxs_physics_attn'
train_steps = 360000
trainer = Trainer1D(
    diffusion,
    dataset=train_dataset,
    dataset_test=test_dataset, # small_val_dataset is to avoid timeout when training on 2 GPUs
    train_batch_size=64,
    train_lr=1e-4,
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
# with open(os.path.join(results_folder, 'norm_coefficients.json'), 'w') as f:
#     json.dump(coefficients, f, indent=4)
# trainer.load(2)
trainer.train(do_profiling=False)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
samples, seqs = trainer.eval_model(test_dataset, batch_size=32) # , cfg_interval_start=0.2

if trainer.accelerator.is_main_process:
    diffusion = trainer.accelerator.unwrap_model(diffusion, keep_torch_compile=True)
    trainer.ema.ema_model.eval()  # Ensure eval mode
    diffusion.eval()
    original_length = train_dataset.max_len
    test_data, test_parameters, *_ = test_dataset[:]
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")

    from cetaceo.evaluators import RegressionEvaluator
    # preds = preds * (cp_max - cp_min) + cp_min
    samples = (samples * train_dataset.p_std) + train_dataset.p_mean
    test_data = (test_data * train_dataset.p_std) + train_dataset.p_mean

    evaluator = RegressionEvaluator()
    print(samples.shape, test_data.shape, test_data[:, :original_length].shape)
    metrics = evaluator(samples, test_data[:, :original_length])
    # metrics = evaluator(preds, cp_test) # cp_test
    evaluator.print_metrics()