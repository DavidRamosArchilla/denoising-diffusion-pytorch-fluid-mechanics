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

model = ViT(
    out_channels=1,
    use_coord_pe=False,
    # num_frequencies=64,
    # coord_dim=2,
    depth=6,
    hidden_size=128,
    patch_size=1,
    num_heads=4,
    input_size=data_sample[0].shape[2],
    cond_dim=data_sample[1].shape[1],
    class_dropout_prob=0,
    in_channels=2,
    learn_sigma=False,
    # use_bias=False,
    use_swiglu=True,
    # use_rope=True,
    # qk_norm=True,
    attn_type="physics",  # window, linear, vanilla, physics
    slice_num=128,
    # window_size=107,
    # num_experts=8,
    # num_experts_per_tok=2
    mlp_ratio=2.5,
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))   

def no_mask_collate(batch):
    pressures, conditions, coords, masks = zip(*batch)
    pressures = torch.stack(pressures)
    conditions = torch.stack(conditions)
    coords = torch.stack(coords)
    masks = torch.stack(masks).unsqueeze(1)
    p_shape = pressures.shape
    pressures = pressures[masks]
    pressures = pressures.view(p_shape[0], p_shape[1], -1)
    c_shape = coords.shape
    coords_mask = masks.expand(-1, coords.shape[1], -1)
    coords = coords[coords_mask]
    coords = coords.view(c_shape[0], c_shape[1], -1)
    # print(coords.shape, pressures.shape, conditions.shape)
    return pressures, conditions, coords, masks


results_folder = 'results/airfrans_good_split/vit_xxs_physics_attn_no_coordPE'
train_steps = 360000
trainer = Trainer1D(
    model,
    dataset=train_dataset,
    dataset_test=test_dataset, # small_val_dataset is to avoid timeout when training on 2 GPUs
    train_batch_size=32,
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
    use_muon=True,
    compile_model=True,
    split_batches=True,
    # dataloader_collate_fn=no_mask_collate
)
# with open(os.path.join(results_folder, 'norm_coefficients.json'), 'w') as f:
#     json.dump(coefficients, f, indent=4)
# trainer.load(2)
trainer.train(do_profiling=False)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
samples, seqs = trainer.eval_model(test_dataset, batch_size=32, pad_output=True) # , cfg_interval_start=0.2

def reconstruct_mask(pressures, padded_len: int) -> torch.BoolTensor:
    """
    Reconstructs the attention mask from the original unpadded dataset tensors.

    Args:
        pressures : List of pressure tensors (n_samples, n_points)
        padded_len : the L dimension of your padded (n_samples, n_points) tensor

    Returns:
        mask : (n_samples, padded_len) bool — True = real point, False = padding
    """
    lengths = torch.tensor([p.shape[0] for p in pressures])  # (n_samples,)
    mask = torch.arange(padded_len).unsqueeze(0) < lengths.unsqueeze(1)  # (n_samples, L)
    return mask

if trainer.accelerator.is_main_process:
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")
    original_length = train_dataset.max_len
    test_data, test_parameters, _, test_masks = test_dataset[:]
    test_masks = test_masks.unsqueeze(1)
    test_data = test_data[test_masks]

    samples_mask = reconstruct_mask(pressures_test, samples.shape[-1])  # (n_samples, L)
    samples = samples.squeeze()[samples_mask]
    samples = (samples * train_dataset.p_std) + train_dataset.p_mean
    test_data = (test_data * train_dataset.p_std) + train_dataset.p_mean
    print(test_data.shape, samples.shape)

    from cetaceo.evaluators import RegressionEvaluator
    evaluator = RegressionEvaluator()
    # print(samples.shape, test_data.shape, test_data[:, :original_length].shape)
    # metrics = evaluator(samples, test_data[:, :original_length])
    metrics = evaluator(samples, test_data)
    # metrics = evaluator(preds, cp_test) # cp_test
    evaluator.print_metrics()