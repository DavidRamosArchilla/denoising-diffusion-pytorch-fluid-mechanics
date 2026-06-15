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
from denoising_diffusion_pytorch.video_dit import DiT_models, DiT#, DiTBlock, DiTCoordPE
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
from denoising_diffusion_pytorch.trainer_hybrid import TrainerHybrid, TrainerCP
import shutil
import os


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
fields_train = (fields_train - fields_train_mean) / fields_train_std
fields_test = (fields_test - fields_train_mean) / fields_train_std

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
target_length = 1704
pad_length = target_length - original_length
print("fields_train shape", fields_train.shape)
print("fields_test shape", fields_test.shape)

fields_train = F.pad(fields_train, (0, pad_length))
fields_test = F.pad(fields_test, (0, pad_length))

train_dataset = TensorDataset(fields_train, conds_train)
test_dataset = TensorDataset(fields_test, conds_test)

model = DiT(
    depth=12,
    hidden_size=256,
    patch_size=4,
    num_frames=fields_train.shape[1],
    num_heads=8,
    input_size=fields_train.shape[-1], # dataset grid size
    cond_dim=1, # number of parameters (alpha, mach)
    class_dropout_prob=0.2,
    in_channels=fields_train.shape[2],
    learn_sigma=False,
    use_swiglu=True,
    # use_rope=True,
    qk_norm=True, # when bf16 training
    attn_type="vanilla",  # window, linear, vanilla, physics
    mlp_ratio=2.5,
)
print("Number trainable of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

sampler = Sampler(transport=create_transport(
    # use_cosine_loss=True,
    # use_lognorm=True
))

diffusion = FlowMatching(
    sampler,
    model,
    input_size=fields_train.shape[-1],
    cond_scale=2,
    num_sampling_steps=100,
    sampling_method="euler",
    # shifted_mu=1.0986
)

results_folder = 'results/cylinder_nvidia/deeper_4'
train_steps = 100000
trainer = Trainer1D(
    diffusion,
    dataset=train_dataset,
    dataset_test=test_dataset, # small_val_dataset is to avoid timeout when training on 2 GPUs
    train_batch_size=4,
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
# trainer.load(5)
trainer.train()
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))

samples, seqs = trainer.eval_model(test_dataset, batch_size=8, use_autocast=True)

if trainer.accelerator.is_main_process:

    test_data, test_parameters = test_dataset.tensors
    torch.save(samples, f"{results_folder}/test_predictions_ema.pt")

    from cetaceo.evaluators import RegressionEvaluator
    # preds = preds * (cp_max - cp_min) + cp_min
    samples = (samples * fields_train_std) + fields_train_mean
    test_data = (test_data * fields_train_std) + fields_train_mean

    evaluator = RegressionEvaluator()
    metrics = evaluator(samples, test_data)
    # metrics = evaluator(preds, cp_test) # cp_test
    evaluator.print_metrics()