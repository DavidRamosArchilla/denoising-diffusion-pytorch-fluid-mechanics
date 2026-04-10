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


data_file = 'data/airfrans/ordered_dataset.pt'
data = torch.load(data_file, weights_only=False)
pressures, coords, conditions = data["pressures"], data["coords"], data["conditions"]

dataset = AirfoilDataset(pressures, coords, conditions)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
data_sample = train_dataset[:]
print(f"Data sample: pressure shape {data_sample[0].shape}, condition shape {data_sample[1].shape}, coords shape {data_sample[2].shape}, mask shape {data_sample[3].shape}")

model = DiTMultiShape(
    depth=8,
    hidden_size=256,
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
    attn_type="vanilla",  # window, linear, vanilla
    # window_size=107,
    # num_experts=8,
    # num_experts_per_tok=2
    mlp_ratio=2.5,
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

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

results_folder = 'results/airfrans/dit_xs_cross_self_attn'
train_steps = 300000
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
# with open(os.path.join(results_folder, 'norm_coefficients.json'), 'w') as f:
#     json.dump(coefficients, f, indent=4)
# trainer.load(2)
trainer.train(do_profiling=False)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))

if trainer.accelerator.is_main_process:
    diffusion = trainer.accelerator.unwrap_model(diffusion, keep_torch_compile=True)
    trainer.ema.ema_model.eval()  # Ensure eval mode
    diffusion.eval()
    original_length = dataset.max_len
    test_data, test_parameters, *_ = val_dataset[:]
    # errors, samples = evaluate_model(
    #     trainer.ema.ema_model, # trainer.ema.ema_model, # diffusion
    #     test_parameters,
    #     test_data,
    #     32,
    #     cond_scale=2
    # )
    # samples = samples[:, :original_length]  # Remove padding if it was added
    # print(f"Final errors:\n{errors}")
    # torch.save(samples, f"{results_folder}/test_predictions_ema.pt")

    # from cetaceo.evaluators import RegressionEvaluator
    # # preds = preds * (cp_max - cp_min) + cp_min
    # samples = (samples * dataset.p_std) + dataset.p_mean
    # test_data = (test_data * dataset.p_std) + dataset.p_mean

    # evaluator = RegressionEvaluator()
    # metrics = evaluator(samples, test_data[:, :original_length])
    # # metrics = evaluator(preds, cp_test) # cp_test
    # evaluator.print_metrics()