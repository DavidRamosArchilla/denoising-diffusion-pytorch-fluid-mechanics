from .dit import DiT, FinalLayer1D, CoordEmbedder
import torch


class ViT(DiT):
    def __init__(self, out_channels, use_coord_pe=True, coord_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_coord_pe = use_coord_pe
        if use_coord_pe:
            assert coord_dim is not None
            self.coord_pe = CoordEmbedder(kwargs["hidden_size"], coord_dim=coord_dim)

        self.out_channels = out_channels
        self.final_layer = FinalLayer1D(kwargs["hidden_size"], self.patch_size, self.out_channels, bias=True)

    def forward(self, x, classes=None, context=None, mask=None, return_loss=True, **kwargs):
        """
        context are the coordinate (inputs in this case)
        """
        true_values = x
        # x = self.x_embedder(context) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        pos_embed = self.pos_embed if not self.use_coord_pe else self.coord_pe(context.permute(0, 2, 1))  # (1, T, D)
        x = self.x_embedder(context) + pos_embed 
        force_drop_ids = kwargs.get("force_drop_ids", None)
        y = self.y_embedder(classes, self.training, force_drop_ids)    # (N, D)
        c = y                                # (N, D)
        for block in self.blocks:
            # x = checkpoint(block, x, c, self.feat_rope, use_reentrant=False)
            x = block(x, c, self.feat_rope, mask)                      # (N, T, D)
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