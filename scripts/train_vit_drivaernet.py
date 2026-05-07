import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from denoising_diffusion_pytorch.continuous_classifier_free_guidance_1d import GaussianDiffusion1D, Trainer1D, Unet1D
from denoising_diffusion_pytorch.dit import DiT_models, DiTCoordPE, DiTMultiShape, DiT, FinalLayer1D, DiTBlock, modulate, CoordEmbedder
from denoising_diffusion_pytorch.transport import create_transport, Sampler, FlowMatching
from data_generation.driveaernet_data import build_datasets, mesh_collate_fn
import shutil
import os
import json
from functools import partial


train_ds, val_ds, test_ds, stats = build_datasets(
    data_root  = '/home/d.ramos/DrivAerNet/PressureVTK',
    split_dir  = '/home/d.ramos/DrivAerNet/splits',
    cache_path = '/home/d.ramos/denoising-diffusion-pytorch-fluid-mechanics/data/mesh_cache.pt',
)

class FinalLayerViT(torch.nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, bias=True):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=bias) # nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        )

    def forward(self, x, c):
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # x = modulate(self.norm_final(x), shift, scale)
        x = self.norm_final(x)
        x = self.linear(x)
        return x

class ViTBlock(DiTBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, c=None, feat_rope=None, mask=None):
        if c is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope, mask=mask)
            x = x.contiguous()
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(self.norm1(x), rope=feat_rope, mask=mask)
            x = x.contiguous()
            x = x + self.mlp(self.norm2(x))
        return x

class ViT(DiT):
    def __init__(self, use_coord_pe=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if use_coord_pe:
            self.use_coord_pe = use_coord_pe
            # 3 IS HARDCODED HERE TOO, 3 for xyz. 1 for ordinal numbers (1, 2, 3, ...).
            self.coord_pe = CoordEmbedder(kwargs["hidden_size"], coord_dim=1)
        # 1 is hardcoded, it is the output channels
        self.out_channels = 1
        self.final_layer = FinalLayerViT(kwargs["hidden_size"], self.patch_size, self.out_channels, bias=True)
        self.blocks = torch.nn.ModuleList(
            [
                ViTBlock(
                    kwargs["hidden_size"],
                    kwargs["num_heads"],
                    mlp_ratio=kwargs["mlp_ratio"],
                    attn_type=kwargs["attn_type"],
                    use_swiglu=kwargs["use_swiglu"],
                    qk_norm=kwargs["qk_norm"],
                )
                for _ in range(kwargs["depth"])
            ]
        )

    def forward(self, x, classes=None, context=None, mask=None, return_loss=True, **kwargs):
        """
        context are the coordinate (inputs in this case)
        """
        true_values = x
        # print(context.shape)
        # pos_embed = self.pos_embed if not self.use_coord_pe else self.coord_pe(context.permute(0, 2, 1))

        x = self.x_embedder(context)  # (N, T, D), where T = H * W / patch_size ** 2
        pe_input = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        # repeat pe_inpu to have batch dimension 
        pe_input = pe_input.expand(x.shape[0], -1).unsqueeze(2)
        pos_embed = self.pos_embed if not self.use_coord_pe else self.coord_pe(pe_input)  # (1, T, D)
        x = x + pos_embed  # (N, T, D)
        force_drop_ids = kwargs.get("force_drop_ids", None)
        c = None
        if classes is not None:
            y = self.y_embedder(classes, self.training, force_drop_ids)    # (N, D)
            c = y                                # (N, D)
        for block in self.blocks:
            # x = checkpoint(block, x, c, self.feat_rope, use_reentrant=False)
            x = block(x, c, self.feat_rope, None)                      # (N, T, D)
        x = self.final_layer(x, c)               # (B, num_patches, patch_size * out_channels)
        x = self.unpatchify(x)                   # (B, out_channels, S)
        if not return_loss:
            return x
        loss = ((true_values - x) ** 2).mean()
        return loss
    
    def sample(self, classes, context=None, mask=None, return_all_steps=False, **model_kwargs):
        self.eval()
        with torch.inference_mode():
            preds = self(None, classes=classes, context=context, mask=mask, return_loss=False, **model_kwargs)  # Run a forward pass to initialize any lazy modules
        return preds

patch_size = 1
dataloader_collate_fn = partial(mesh_collate_fn, pad_multiple=patch_size)
model = ViT(
    depth=12,
    hidden_size=384,
    patch_size=patch_size,
    num_heads=6,
    # input_size=data_sample[0].shape[2],
    # cond_dim=data_sample[1].shape[1],
    class_dropout_prob=0,
    in_channels=3, # xyz
    learn_sigma=False,
    # use_bias=False,
    use_swiglu=True,
    # use_rope=True,
    qk_norm=True,
    attn_type="linear",  # window, linear, vanilla
    # window_size=107,
    # num_experts=8,
    # num_experts_per_tok=2
    mlp_ratio=2.5,
)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))   
print("Number trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))   


results_folder = 'results/drivaernet/vit_first'
train_steps = 300000
trainer = Trainer1D(
    model,
    dataset=train_ds,
    # dataset_test=val_ds, # small_val_dataset is to avoid timeout when training on 2 GPUs
    train_batch_size=1,
    train_lr=2e-4,
    num_samples=9,
    train_num_steps=train_steps,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,     # turn on mixed precision
    mixed_precision_type='bf16',
    results_folder=results_folder,  # folder to save results to
    save_and_sample_every=20000,
    eta_min_scheduler=4e-6,
    max_grad_norm=1.0,
    # use_cpu=True,
    # use_muon=True,
    # compile_model=True,
    split_batches=True,
    dataloader_collate_fn=dataloader_collate_fn
)
# with open(os.path.join(results_folder, 'norm_coefficients.json'), 'w') as f:
#     json.dump(coefficients, f, indent=4)
# trainer.load(2)
trainer.train(do_profiling=False)
shutil.copy(__file__, os.path.join(results_folder, os.path.basename(__file__)))
samples, seqs = trainer.eval_model(test_ds, batch_size=32) # , cfg_interval_start=0.2
