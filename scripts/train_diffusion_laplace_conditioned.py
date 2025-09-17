import numpy as np
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch.continuous_classifier_free_guidance import Unet, GaussianDiffusion, Trainer
import shutil
import os

data = np.load("data/laplace/laplace_dataset_solutions.npy")
# at the moment the model expects inputs in [0, 1], like grayscale images 
data = (data - data.min()) / (data.max() - data.min())  # scale to [0, 1]
# data = data * 2 - 1  # scale
data = data[:, np.newaxis, :, :]  # add channel dimension   
parameters = np.load("data/laplace/laplace_dataset_parameters.npy")
# normalize parameters to [0, 1]
parameters = (parameters - parameters.min(axis=0)) / (parameters.max(axis=0) - parameters.min(axis=0))
print(f"max parameter 1: {parameters[:,0].max()}, min parameter 1: {parameters[:,0].min()}")
print(f"max parameter 2: {parameters[:,1].max()}, min parameter 2: {parameters[:,1].min()}")

dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(parameters, dtype=torch.float32))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4),#, 8),
    # flash_attn = False,
    channels = 1,
    cond_dim=2,
    # full_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = (64, 64),
    objective = 'pred_noise',  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=50,
    timesteps = 1000,    # number of steps
)


print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 1e9} GB")

results_folder = 'results/laplace_conditioned_small_normalized_asdasd'

trainer = Trainer(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset,
    train_batch_size = 32,
    train_lr = 8e-5,
    num_samples=9,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    results_folder = results_folder,  # folder to save results to
    save_and_sample_every=2000,
    augment_horizontal_flip=False,
    # use_cpu=True
)

# save this script in the results folder for reference
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))

trainer.train()
torch.cuda.empty_cache()  # Clear GPU memory
trainer.ema.ema_model.eval()  # Ensure eval mode
diffusion.eval()

with torch.inference_mode():
    sampled_seq = diffusion.sample(classes=torch.tensor(parameters[800:]))  # Original model
    torch.save(sampled_seq, results_folder + '/sampled_seq.pt')
    # sampled_seq = trainer.ema.ema_model.sample(batch_size=64)  # Use EMA model
    # torch.save(sampled_seq, results_folder + '/sampled_seq_ema.pt')
