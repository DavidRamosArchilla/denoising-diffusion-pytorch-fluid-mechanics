import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from tqdm import tqdm

from denoising_diffusion_pytorch.graph_unet_cfg_diffusion import (
    ConditionedGraphUNet,
    GraphDiffusion,
    Trainer,
)

class AeronefDataset(Dataset):
    def __init__(self, data_directory, target_field, num_points=None, coef_norm=None):
        super(AeronefDataset, self).__init__()
       
        self.graph_dataset = []
        self.conditions = []
        self.coef_norm = coef_norm
        self.num_points = num_points

        print("Processing dataset...")
        self.process_data(data_directory, target_field)
        
    def process_data(self, data_directory, target_field):

        print('Loading raw data')
        db_random = np.load(os.path.join(data_directory, 'db_random.npy'), allow_pickle=True).item()
        # db_cyc = np.load(os.path.join(data_directory, 'db_cyc.npy'), allow_pickle=True).item()
        db = db_random
        # Merge db_random and db_cyc
        # db = {key: np.concatenate((db_random[key], db_cyc[key]), axis=0) for key in ['Pressure','Xcoordinate','Ycoordinate','Vinf','Alpha','idx']}
        print('Raw data Loaded, normalizing data')
    

        self.coef_norm = {'mean_in': None, 'std_in': None,'min_out': None, 'max_out': None, 'min': None, 'max': None} 
        min_out = db[target_field].min()
        max_out = db[target_field].max()
        db[target_field] = (db[target_field] - min_out) / (max_out - min_out)
        self.coef_norm['min_out'] = min_out
        self.coef_norm['max_out'] = max_out

         # Normalize condition data (Vinf and Alpha)
        cond_data = np.stack([db['Vinf'], db['Alpha']], axis=1)
        mean_in = cond_data.mean(axis=0)
        std_in = cond_data.std(axis=0)
        cond_data = (cond_data - mean_in) / std_in
        self.coef_norm['mean_in'] = mean_in
        self.coef_norm['std_in'] = std_in
        total_points = db['Xcoordinate'].shape[1]
        if self.num_points is not None:
            subsample_indices = np.random.choice(total_points, self.num_points, replace=False)
            
        for idx in tqdm(range(len(db['idx']))):
            X_coord = db['Xcoordinate'][idx] - db['Xcoordinate'][idx].min() / (db['Xcoordinate'][idx].max() - db['Xcoordinate'][idx].min())
            Y_coord = db['Ycoordinate'][idx] - db['Ycoordinate'][idx].min() / (db['Ycoordinate'][idx].max() - db['Ycoordinate'][idx].min())
            pos = torch.tensor(np.stack((X_coord, Y_coord), axis=1), dtype=torch.float)
            output = torch.tensor(db[target_field][idx], dtype=torch.float).unsqueeze(-1)
            cond = torch.tensor(cond_data[idx], dtype=torch.float)
            # cond = cond.repeat(pos.shape[0], 1)  # Repeat cond to match pos size

            if self.num_points is not None and pos.shape[0] > self.num_points:
                # subsample_indices = np.random.choice(pos.shape[0], self.num_points, replace=False)
                pos = pos[subsample_indices]
                output = output[subsample_indices]
                # cond = cond[subsample_indices]

            # Create edges using k-nearest neighbors or radius graph
            edge_index = knn_graph(pos, k=8, batch=None, loop=False)
            # OR: edge_index = radius_graph(pos, r=0.1, batch=None, loop=False)
            # self.graph_dataset.append(Data(x=x, pos=pos, y=output))#, edge_index=edge_index))
            input_data = torch.cat([pos, output], dim=1)
            self.graph_dataset.append(Data(x=input_data, pos=pos, edge_index=edge_index))
            self.conditions.append(cond) 
    
    
    def create_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        if seed is not None:
            np.random.seed(seed)
        
        total_samples = len(self.graph_dataset)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        train_end = int(train_ratio * total_samples)
        val_end = train_end + int(val_ratio * total_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        self.train_dataset = [(self.graph_dataset[i], self.conditions[i]) for i in train_indices]
        self.val_dataset = [(self.graph_dataset[i], self.conditions[i]) for i in val_indices]
        self.test_dataset = [(self.graph_dataset[i], self.conditions[i]) for i in test_indices]


    def __getitem__(self, index):
        return self.graph_dataset[index], self.conditions[index]

    def __len__(self):
        return len(self.graph_dataset)

data_directory = 'data/aeronef/'
train_dataset_obj = AeronefDataset(data_directory, "Pressure", num_points=10000)
train_dataset_obj.create_splits(train_ratio=0.8, val_ratio=0.2, test_ratio=1e9, seed=42)

example_graph = train_dataset_obj.graph_dataset[100]
print(example_graph)
model = ConditionedGraphUNet(
    dim=32,
    in_channels=3,
    out_channels=3,
    cond_dim=2,
    cond_drop_prob=0.0,
    dim_mults=(1, 2, 4),
    pool_ratios=0.5,
    sum_res=False,
    act='relu',
)

diffusion = GraphDiffusion(
    model,
    num_mesh_points=example_graph.x.shape[0],
    default_mesh_connectivity=example_graph.edge_index,
    objective="pred_noise",  # 'pred_noise' or 'pred_x0'
    beta_schedule="cosine",
    sampling_timesteps=1000,
    timesteps=1000,  # number of steps
    min_snr_loss_weight=True,
    min_snr_gamma=5,
)

results_folder = 'results/aeronef/gunet_M'
train_steps = 100000

trainer = Trainer(
    diffusion,
    dataset=train_dataset_obj.train_dataset,
    train_batch_size=32,
    train_lr=8e-5,
    num_samples=9,
    train_num_steps=train_steps+4,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=10000,
    # use_cpu=True
)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
trainer.train()
trainer.ema.ema_model.eval()  # Ensure eval mode
diffusion = trainer.accelerator.unwrap_model(diffusion)
diffusion.eval()
test_dataset = train_dataset_obj.val_dataset