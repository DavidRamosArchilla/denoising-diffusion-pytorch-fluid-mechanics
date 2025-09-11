from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import TensorDataset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from scipy.ndimage import zoom
import shutil
import os


class TensorDatasetWrapper(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)
    
    def __getitem__(self, index):
        return super().__getitem__(index)[0]

data = loadmat("data/cylinder_nektar_wake.mat")

presiones = data['p_star'].reshape(50, 100, -1).transpose(2, 0, 1)
zoom_factors = (1, 64/50, 128/100)  # (1, 1.28, 1.28), el 1 es para tenerlo vectorizado en la fimension de los samples
presiones = zoom(presiones, zoom_factors, order=1)  # downsample in x direction

# add channel dimension
presiones = presiones[:, np.newaxis, :, :]
print(f"Presiones shape: {presiones.shape}")

dataset = TensorDatasetWrapper(torch.tensor(presiones, dtype=torch.float32))

model = Unet(
    dim = 128,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
    channels = 1,  
)

diffusion = GaussianDiffusion(
    model,
    image_size = (64, 128),
    objective = 'pred_noise',  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=50,
    timesteps = 500    # number of steps
)

# torch.set_float32_matmul_precision('high')
# diffusion = torch.compile(diffusion)
# print("Model and diffusion compiled")

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Model size estimate: {sum(p.numel() for p in model.parameters()) * 4 / 1e9} GB")

results_folder = 'results_2d_test_sampling'

trainer = Trainer(
    diffusion,
    # 'path/to/your/images',
    dataset=dataset,
    train_batch_size = 8,
    train_lr = 8e-5,
    num_samples=9,
    train_num_steps = 16000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    results_folder = results_folder,  # folder to save results to
    save_and_sample_every=2000,
    augment_horizontal_flip=False
)

# save this script in the results folder for reference
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))

trainer.train()

trainer.load(8)

torch.cuda.empty_cache()  # Clear GPU memory
trainer.ema.ema_model.eval()  # Ensure eval mode
diffusion.eval()

with torch.inference_mode():
    sampled_seq = diffusion.sample(batch_size=64)  # Original model
    torch.save(sampled_seq, results_folder + '/sampled_seq.pt')
    sampled_seq = trainer.ema.ema_model.sample(batch_size=64)  # Use EMA model
    torch.save(sampled_seq, results_folder + '/sampled_seq_ema.pt')
