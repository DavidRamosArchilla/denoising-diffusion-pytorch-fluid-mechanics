"""
Ring Attention built on Flash Attention 4 (flash_attn.cute).

FA4 does not expose return_softmax_lse, so we compute LSE separately via a
chunked logsumexp over QK^T. This is a single extra pass over Q and K (no V),
O(N_local * chunk_size) memory, and negligible cost vs the full attention.

Public API
----------
ring_flash_attn_fa4(q, k, v, group, dropout_p, causal) -> out
    q, k, v : (B, N_local, heads, head_dim)
    group   : dist.ProcessGroup  (the CP process group)
    returns : (B, N_local, heads, head_dim)
"""

import torch
import torch.distributed as dist
try:
    from flash_attn.cute import flash_attn_func
    _is_fa4_available = True
except ImportError as e:
    _is_fa4_available = False
    print(f"Error importing flash_attn.cute: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LSE helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lse(
    q: torch.Tensor,          # (B, N_q, H, D)
    k: torch.Tensor,          # (B, N_k, H, D)
    scale: float | None = None,
    chunk_size: int = 4096,
) -> torch.Tensor:             # (B, H, N_q)  — float32
    """
    Compute log-sum-exp of attention logits without materialising the full
    N_q × N_k score matrix.  Processes K in chunks of `chunk_size` tokens and
    accumulates via logaddexp, so peak memory is O(N_q * chunk_size).
    """
    B, Nq, H, D = q.shape
    if scale is None:
        scale = D ** -0.5

    # accumulate in fp32 for numerical stability
    q_f = q.float()
    lse = torch.full((B, H, Nq), float('-inf'), device=q.device, dtype=torch.float32)

    for start in range(0, k.shape[1], chunk_size):
        k_chunk = k[:, start : start + chunk_size].float()      # (B, C, H, D)
        # scores: (B, H, N_q, C)
        scores = torch.einsum('bqhd,bchd->bhqc', q_f, k_chunk) * scale
        chunk_lse = torch.logsumexp(scores, dim=-1)             # (B, H, N_q)
        lse = torch.logaddexp(lse, chunk_lse)

    return lse                                                   # (B, H, N_q)


# ─────────────────────────────────────────────────────────────────────────────
# Online-softmax combiner
# ─────────────────────────────────────────────────────────────────────────────

def _combine(
    out_acc: torch.Tensor,   # (B, N, H, D)
    lse_acc: torch.Tensor,   # (B, H, N)
    out_new: torch.Tensor,   # (B, N, H, D)
    lse_new: torch.Tensor,   # (B, H, N)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge a new partial attention result into the running accumulator.

    lse  = logaddexp(lse_acc, lse_new)
    out  = exp(lse_acc - lse) * out_acc  +  exp(lse_new - lse) * out_new
    """
    lse = torch.logaddexp(lse_acc, lse_new)          # (B, H, N)

    # (B, H, N) → (B, N, H, 1) for broadcasting with (B, N, H, D)
    def _w(l):
        return l.permute(0, 2, 1).unsqueeze(-1)

    w_acc = torch.exp(_w(lse_acc) - _w(lse))
    w_new = torch.exp(_w(lse_new) - _w(lse))

    out = w_acc * out_acc + w_new * out_new
    return out, lse


# ─────────────────────────────────────────────────────────────────────────────
# P2P ring communication
# ─────────────────────────────────────────────────────────────────────────────

def _ring_send_recv(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Send to rank+1, receive from rank-1 (one ring step)."""
    rank      = dist.get_rank(group)
    cp_size   = dist.get_world_size(group)
    all_ranks = dist.get_process_group_ranks(group)
    send_to   = all_ranks[(rank + 1) % cp_size]
    recv_from = all_ranks[(rank - 1) % cp_size]

    recv_buf = torch.empty_like(tensor)
    ops = [
        dist.P2POp(dist.isend, tensor,   send_to,   group),
        dist.P2POp(dist.irecv, recv_buf, recv_from, group),
    ]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return recv_buf


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ring_flash_attn_fa4(
    q:          torch.Tensor,
    k:          torch.Tensor,
    v:          torch.Tensor,
    group:      dist.ProcessGroup,
    dropout_p:  float = 0.,
    causal:     bool  = False,
    lse_chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Ring attention using FA4 for local chunk computation.

    FA4 handles the within-rank attention efficiently; we layer the cross-rank
    combination on top using a manually computed LSE.

    Args:
        q, k, v        : (B, N_local, heads, head_dim) — this rank's sequence shard
        group          : CP process group
        lse_chunk_size : chunk size for the LSE computation (trades memory vs speed)

    Returns:
        out : (B, N_local, heads, head_dim)
    """
    assert not causal, "Diffusion transformers use bidirectional attention; set causal=False."

    cp_size = dist.get_world_size(group)
    if cp_size == 1:
        if not _is_fa4_available:
            # fall to sdpa
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=causal,
            )
        out, *_ = flash_attn_func(q, k, v, causal=False)
        return out

    scale = q.shape[-1] ** -0.5

    k_buf = k.contiguous()
    v_buf = v.contiguous()

    out_acc: torch.Tensor | None = None
    lse_acc: torch.Tensor | None = None

    for step in range(cp_size):
        # ── 1. Local attention output via FA4 ────────────────────────────────
        out_step, *_ = flash_attn_func(
            q, k_buf, v_buf,
            causal=False,
        )                                             # (B, N, H, D)

        # ── 2. LSE for this KV chunk (cheap chunked logsumexp, no V needed) ─
        lse_step = _compute_lse(q, k_buf, scale=scale, chunk_size=lse_chunk_size)
        # (B, H, N)

        # ── 3. Merge into accumulator ────────────────────────────────────────
        if out_acc is None:
            out_acc = out_step.float()
            lse_acc = lse_step
        else:
            out_acc, lse_acc = _combine(out_acc, lse_acc, out_step.float(), lse_step)

        # ── 4. Rotate KV (skip on last step) ─────────────────────────────────
        if step < cp_size - 1:
            kv      = torch.stack([k_buf, v_buf], dim=0)   # (2, B, N, H, D)
            kv_recv = _ring_send_recv(kv, group)
            k_buf, v_buf = kv_recv[0], kv_recv[1]

    return out_acc.to(q.dtype)