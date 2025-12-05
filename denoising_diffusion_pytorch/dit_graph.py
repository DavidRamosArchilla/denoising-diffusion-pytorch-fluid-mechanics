import torch
import torch.nn as nn
import numpy as np
# from torch_scatter import scatter_mean # Requires: pip install torch-scatter

from .dit import DiTBlock, ConditionEmbedder, TimestepEmbedder


def scatter_mean_pure_torch(src, index, dim=0, dim_size=None):
    """
    Pure PyTorch implementation of scatter_mean.
    
    Args:
        src: Source tensor to scatter
        index: Index tensor indicating where to scatter each element
        dim: Dimension along which to scatter
        dim_size: Size of output dimension (if None, inferred from index.max() + 1)
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Create output tensor for sum
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    
    # Create count tensor to track how many elements per index
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    
    # Scatter add to accumulate values
    out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
    
    # Count occurrences of each index
    ones = torch.ones_like(index, dtype=src.dtype)
    count.scatter_add_(0, index, ones)
    
    # Avoid division by zero
    count = count.clamp(min=1)
    
    # Compute mean by dividing by count
    # Reshape count to broadcast correctly
    count_shape = [1] * len(out.shape)
    count_shape[dim] = dim_size
    count = count.view(count_shape)
    
    return out / count


class GeometricPatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, bias=True):
        super().__init__()
        # MLP to process individual nodes before aggregation
        # Input: Pressure (1) + Relative Pos (2) = 3
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x, cluster_indices, relative_pos):
        """
        x: (B, 1, N_total) - The pressure field
        cluster_indices: (N_total,) - Which patch each node belongs to (0 to K-1)
        relative_pos: (N_total, 2) - Node pos relative to its patch centroid
        """
        B, C, N = x.shape
        
        # 1. Prepare inputs: Combine physics (x) with geometry (pos)
        # Transpose to (B, N, C)
        x = x.permute(0, 2, 1) 
        
        # Expand relative_pos to batch dimension
        rel_pos_batch = relative_pos.unsqueeze(0).expand(B, -1, -1) # (B, N, 2)
        
        # Concatenate: (B, N, C+2)
        node_features = torch.cat([x, rel_pos_batch], dim=-1)
        
        # 2. Apply PointNet MLP per node
        node_embeddings = self.node_mlp(node_features) # (B, N, Hidden)
        
        # 3. POOLING (The "Patching" step)
        # We average all nodes belonging to the same patch
        # Output shape: (B, Num_Patches, Hidden)
        
        # Flatten batch for scatter: (B*N, Hidden)
        flat_embeddings = node_embeddings.view(-1, node_embeddings.shape[-1])
        
        # Repeat cluster indices for the batch: (B*N,)
        batched_cluster_indices = cluster_indices.repeat(B)
        # Adjust indices so batch 1 doesn't mix with batch 0
        num_patches = cluster_indices.max() + 1
        batch_offsets = torch.arange(B, device=x.device).repeat_interleave(N) * num_patches
        final_indices = batched_cluster_indices + batch_offsets

        # Scatter Mean: Aggregates variable number of nodes into fixed patches
        patch_embeddings = scatter_mean_pure_torch(
            flat_embeddings, 
            final_indices, 
            dim=0, 
            dim_size=B * num_patches
        )
        
        # Reshape back to (B, Num_Patches, Hidden)
        patch_embeddings = patch_embeddings.view(B, num_patches, -1)
        
        return self.proj(patch_embeddings)

class GeometricUnpatchify(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        
        self.norm = nn.RMSNorm(hidden_size)
        
        # MLP to decode: Takes Patch Token + Relative Pos -> Pressure
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, out_channels)
        )

    def forward(self, x, cluster_indices, relative_pos):
        """
        x: (B, Num_Patches, Hidden) - Output from DiT Blocks
        cluster_indices: (N_total,)
        relative_pos: (N_total, 2)
        """
        B, K, D = x.shape
        N_total = len(cluster_indices)
        
        # 1. Norm
        x = self.norm(x)
        
        # 2. BROADCAST (The inverse of Pooling)
        # We map the patch token back to every node in that patch
        # x[b, i, :] goes to all nodes where cluster_index == i
        
        # Gather: (B, N_total, Hidden)
        # We expand x to match cluster_indices
        # cluster_indices needs shape (1, N_total, 1) expanded to (B, N_total, D)
        expanded_indices = cluster_indices.view(1, -1, 1).expand(B, N_total, D)
        node_tokens = torch.gather(x, 1, expanded_indices)
        
        # 3. Refine with Geometry
        # Concatenate the coarse patch info with fine-grained relative positions
        rel_pos_batch = relative_pos.unsqueeze(0).expand(B, -1, -1) # (B, N, 2)
        
        decode_input = torch.cat([node_tokens, rel_pos_batch], dim=-1) # (B, N, Hidden+2)
        
        # 4. Predict Pressure
        out = self.decoder_mlp(decode_input) # (B, N, Out_Channels)
        
        return out.permute(0, 2, 1) # Return (B, C, N)
    
class FourierEmbedder2D(nn.Module):
    def __init__(self, hidden_size, num_frequencies=32, scale=1.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.hidden_size = hidden_size
        self.scale = scale
        
        # 1. Random Gaussian matrix to project (x,y,z) into random frequencies
        # Shape: (2, num_frequencies)
        self.freq_weights = nn.Parameter(torch.randn(2, num_frequencies) * scale, requires_grad=False)
        
        # 2. MLP to map the concatenated sin/cos features to the DiT hidden size
        # Input dim: num_frequencies * 2 (sin + cos)
        self.mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        """
        x: (..., 2) coordinates (e.g., centroids)
        """
        # x shape: (Batch, Num_Patches, 2) or (Num_Patches, 2)
        
        # 1. Project coords: (..., 2) @ (2, F) -> (..., F)
        # 2 * pi ensures the weights act as frequencies
        args = (x @ self.freq_weights) * 2 * np.pi
        
        # 2. Compute Fourier features
        # Cat sin and cos: (..., F) -> (..., 2*F)
        fourier_features = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # 3. Project to model dimension
        return self.mlp(fourier_features)


class GraphDiT(nn.Module):
    def __init__(
        self,
        mesh_pos,
        num_patches=64,
        in_channels=1,
        cond_dim=2,
        class_dropout_prob=0.1,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=True,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.cond_dim = cond_dim

        self.self_condition = False
        # 1. Precompute Clusters (Done once)
        # cluster_ids: (N_total,)
        # rel_pos: (N_total, 2)
        # centroids: (Num_Patches, 2)
        self.cluster_ids, self.rel_pos, centroids = self._init_clusters(mesh_pos, num_patches)
        
        # Register as buffers so they move to GPU with model but aren't trained
        self.register_buffer('cluster_indices', self.cluster_ids)
        self.register_buffer('relative_positions', self.rel_pos)
        
        # 2. New Embedder
        self.x_embedder = GeometricPatchEmbed(in_channels=1, hidden_size=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ConditionEmbedder(cond_dim, hidden_size, class_dropout_prob)
        # 3. Pos Embed (Based on Centroids)
        # Use Fourier features on self.centroids
        self.pos_embed = FourierEmbedder2D(hidden_size)
        self.register_buffer("centroids", centroids)
        # 4. Transformer Blocks (Unchanged)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # 5. New Decoder (Replaces FinalLayer)
        self.final_layer = GeometricUnpatchify(hidden_size, out_channels=1)

    def _init_clusters(self, pos, k):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(pos)
        
        labels = torch.tensor(kmeans.labels_, dtype=torch.long)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
        
        # Calculate relative position for every node: Node_Pos - Centroid_of_its_patch
        # Gather centroids to node positions
        node_centroids = centroids[labels]
        rel_pos = torch.tensor(pos, dtype=torch.float) - node_centroids
        
        return labels, rel_pos, centroids

    def forward(self, x, t, y, *args, **kwargs):
        # x: (B, 1, N_points)
        
        # Embed: Graph -> Transformer Tokens
        x = self.x_embedder(x, self.cluster_indices, self.relative_positions) 
        # x is now (B, Num_Patches, Hidden) - Structured!
        # x = x + self.pos_embed
        # x = self.pos_embed(x)
        pos = self.pos_embed(self.centroids) 
        t = self.t_embedder(t)   
        force_drop_ids = kwargs["force_drop_ids"] if "force_drop_ids" in kwargs else None
        y = self.y_embedder(y, self.training, force_drop_ids)    # (N, D)
        # TODO: probar a concatenar en vez de sumar (habra que ajustar las dimensiones)
        c = t + y    
        # ... Transformer Blocks Loop ...
        for block in self.blocks:
            x = block(x, c)
        # Decode: Transformer Tokens -> Graph
        x = self.final_layer(x, self.cluster_indices, self.relative_positions)
        # x is now (B, 1, N_points)
        return x
    
    def forward_with_cond_scale(self, x, t, y, cond_scale, *args, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
        batch_size = x.shape[0]
        # TODO: esta aproximacion no va a ir porque cluster indeices solo esta una vez 
        combined = torch.cat([x, x], dim=0)
        force_drop_ids = torch.cat(
            [
                torch.zeros((batch_size,), dtype=torch.bool, device=x.device),
                torch.ones((batch_size,), dtype=torch.bool, device=x.device),
            ],
            dim=0,
        )
        y_combined = torch.cat([y, y], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        model_out = self.forward(combined, t_combined, y_combined, force_drop_ids=force_drop_ids)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # separate noise predictions and from variance predictions if present
        eps, rest = model_out[:, :self.channels], model_out[:, self.channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
        return half_eps, uncond_eps