import pyLOM
import pyLOM.NN
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import os
import math
from denoising_diffusion_pytorch.dit import MoeMLP, AddAuxiliaryLoss


data = np.load("data/aeronef/db_random.npy", allow_pickle=True).item()

x_coord = data['Xcoordinate']
y_coord = data['Ycoordinate']
spatial_coords = np.stack((x_coord, y_coord), axis=-1)
cp = data['Cp']

# normalize pressure to zero mean and unit variance
cp_mean, cp_std = cp.mean(), cp.std()
cp = (cp - cp_mean) / cp_std
print(cp.shape, y_coord.shape, spatial_coords.shape)
print(cp.shape, spatial_coords.shape, y_coord.shape)

vel_inf = data['Vinf']
alpha = data['Alpha']
vel_mean, vel_std = vel_inf.mean(), vel_inf.std()
alpha_mean, alpha_std = alpha.mean(), alpha.std()
vel_inf = (vel_inf - vel_mean) / vel_std
alpha = (alpha - alpha_mean) / alpha_std
vel_inf = vel_inf.tolist()
alpha = alpha.tolist()

dataset = pyLOM.NN.Dataset(
    variables_out=(cp,),
    variables_in=data["Airfoil"],
    parameters=(vel_inf, alpha),
    snapshots_by_column=False
)

class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, h = hidden_states.shape    
        # print(bsz, seq_len, h)    
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, aux_topk, device=hidden_states.device)).div_(aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class MoE(pyLOM.NN.MLP, torch.nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, embed_dim, in_dim, out_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2, pretraining_tp=1, n_shared_experts=0, device='cpu'):
        # super().__init__()
        # call Module.__init__  to avoid MLP init
        torch.nn.Module.__init__(self)        
        self.input_layer = torch.nn.Linear(in_dim, embed_dim)
        self.norm = nn.RMSNorm(embed_dim)
        self.output_layer = torch.nn.Linear(embed_dim, out_dim)
        self.output_size = out_dim # to keep the interface with pyLOM.NN.MLP
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = torch.nn.ModuleList([MoeMLP(hidden_size=embed_dim, intermediate_size=int(mlp_ratio * embed_dim), pretraining_tp=pretraining_tp) for i in range(num_experts)])
        self.gate = MoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        # hardcoded. This makes that the tokens go through shared experts always. This should be great
        self.n_shared_experts = n_shared_experts
        self.device = device
        if self.n_shared_experts > 0:
            intermediate_size =  embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size=embed_dim, intermediate_size=intermediate_size, pretraining_tp=pretraining_tp)
        
        self.to(device)
    
    def forward(self, hidden_states):
        hidden_states = self.input_layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states) 
        # print(topk_idx.tolist(), print(len(topk_idx.tolist()))) 
        # global selected_ids_list
        # selected_ids_list.append(topk_idx.tolist())

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts): 
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).float()
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        y = self.output_layer(y)
        return y
    

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            
            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, in_dim, out_dim, embed_dim=256, mlp_ratio=4, num_experts=16, num_experts_per_tok=2, pretraining_tp=1, device='cpu'):
        model = cls(
            in_dim=in_dim,
            out_dim=out_dim,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,    
            pretraining_tp=pretraining_tp,
            device=device
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model

dataset_train, dataset_test = dataset.get_splits_by_parameters([0.8, 0.2])

sample_input, sample_output = dataset_train[0]
# 128 expertos, 8 veces mas pequeños, 4 shared y 12 activos seria la configuracion de deepseek
model = MoE(
    in_dim=sample_input.shape[0],
    out_dim=sample_output.shape[0],
    embed_dim=1024 // 8,
    mlp_ratio=4,
    num_experts=128, 
    num_experts_per_tok=12,
    device="cuda",
    n_shared_experts=4,
)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

training_params = {
    "epochs": 1500,
    "lr": 2e-4,
    "lr_gamma": 0.95,
    "lr_scheduler_step": 25, # in epochs, so every 75 epochs the lr is multiplied by lr_gamma (0.99)
    "batch_size": 2**14,
    "loss_fn": torch.nn.MSELoss(),
    "optimizer_class": torch.optim.AdamW,
    "print_rate_epoch": 1,
    "print_rate_batch": 50,
    # "pin_memory":False,
    "num_workers":8,
}

pipeline = pyLOM.NN.Pipeline(
    train_dataset=dataset_train,
    test_dataset=dataset_test,
    model=model,
    training_params=training_params,
)

results_dir = "results/aeronef_cp/moe_128exp_12tok_4shared"
os.makedirs(results_dir, exist_ok=True)
training_logs = pipeline.run()
model.save(f'{results_dir}/moe_model.pt')
# model = model.load(f'{results_dir}/moe_model.pt', in_dim=sample_input.shape[0], out_dim=sample_output.shape[0], device='cuda')
preds = model.predict(dataset_test, batch_size=2**14, return_targets=False)
preds = (preds * cp_std) + cp_mean
y_true = dataset_test[:][1]
y_true = (y_true * cp_std) + cp_mean
evaluator = pyLOM.NN.RegressionEvaluator()
evaluator(y_true, preds)
evaluator.print_metrics()

def true_vs_pred_plot(y_true, y_pred, path):
    """
    Auxiliary function to plot the true vs predicted values
    """
    num_plots = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):
        plt.subplot(num_plots, 1, j + 1)
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.5)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"Scatterplot for Component {j+1}")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=300)

def plot_train_test_loss(train_loss, test_loss, path):
    """
    Auxiliary function to plot the training and test loss
    """
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    total_epochs = len(test_loss) # test loss is calculated at the end of each epoch
    total_iters = len(train_loss) # train loss is calculated at the end of each iteration/batch
    iters_per_epoch = total_iters // total_epochs
    plt.plot(np.arange(iters_per_epoch, total_iters+1, step=iters_per_epoch), test_loss, label="Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=300)


true_vs_pred_plot(y_true, preds, f'{results_dir}/true_vs_pred.png')
plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], f'{results_dir}/train_test_loss.png')