# Copyright 2026 Linum Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File: attention.py
Description: Attention implementations with variable sequence length support for the DiT model.
             Supports Flash Attention 3 with automatic fallback to PyTorch SDPA.
"""
import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_interface
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# ----------------------------------------------------------------------------
# ATTENTION DISPATCHER
# ----------------------------------------------------------------------------
def compute_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_q: torch.Tensor,
    mask_k: torch.Tensor,
) -> torch.Tensor:
    """
    Compute attention with variable sequence length support.

    Automatically dispatches to Flash Attention 3 if available, otherwise
    falls back to PyTorch's scaled_dot_product_attention.

    Args:
        q (torch.Tensor):
            Query tensor of shape [B, L1, num_heads, head_dim]
        k (torch.Tensor):
            Key tensor of shape [B, L2, num_heads, head_dim]
        v (torch.Tensor):
            Value tensor of shape [B, L2, num_heads, head_dim]
        mask_q (torch.Tensor):
            Boolean mask for valid query positions, shape [B, L1].
            True indicates valid positions.
        mask_k (torch.Tensor):
            Boolean mask for valid key/value positions, shape [B, L2].
            True indicates valid positions.

    Returns:
        torch.Tensor:
            Output tensor of shape [B, L1, num_heads, head_dim]
    """
    if FLASH_ATTN_AVAILABLE:
        return flash_attention(q, k, v, mask_q, mask_k)
    else:
        return sdpa_attention(q, k, v, mask_q, mask_k)


# ----------------------------------------------------------------------------
# FLASH ATTENTION 3 IMPLEMENTATION
# ----------------------------------------------------------------------------
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_q: torch.Tensor,
    mask_k: torch.Tensor,
) -> torch.Tensor:
    """
    Flash Attention 3 with variable sequence length support.

    Uses FlashAttention 3's varlen API to efficiently handle batches where
    each sample may have different valid sequence lengths. Sequences are
    packed (removing padding) before attention and unpacked afterward.

    Optimized for torch compile.

    Args:
        q (torch.Tensor):
            Query tensor of shape [B, L1, num_heads, head_dim]
        k (torch.Tensor):
            Key tensor of shape [B, L2, num_heads, head_dim]
        v (torch.Tensor):
            Value tensor of shape [B, L2, num_heads, head_dim]
        mask_q (torch.Tensor):
            Boolean mask for valid query positions, shape [B, L1].
            True indicates valid positions.
        mask_k (torch.Tensor):
            Boolean mask for valid key/value positions, shape [B, L2].
            True indicates valid positions.

    Returns:
        torch.Tensor:
            Output tensor of shape [B, L1, num_heads, head_dim]
    """
    b, seq_len_padded, out_dtype = q.size(0), q.size(1), q.dtype
    num_heads = q.shape[2]
    head_dim = q.shape[3]

    def to_bfloat16(x):
        return x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)

    # Pack sequences: remove padding to create dense tensors for flash attention.
    # FlashAttention's varlen API expects concatenated sequences without padding.
    q_packed = pack_tensor(q, mask_q)
    seq_lens_q = mask_q.sum(dim=1).to(torch.int32)
    q = to_bfloat16(q_packed)

    k_packed = pack_tensor(k, mask_k)
    v_packed = pack_tensor(v, mask_k)
    seq_lens_k = mask_k.sum(dim=1).to(torch.int32)

    k = to_bfloat16(k_packed)
    v = to_bfloat16(v_packed)

    # Compute cumulative sequence lengths for FlashAttention's varlen API.
    # cu_seqlens format: [0, len_0, len_0+len_1, ...] for indexing into packed tensors.
    x_packed = flash_attn_interface.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.cat([seq_lens_q.new_zeros([1]), seq_lens_q
                                ]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
        cu_seqlens_k=torch.cat([seq_lens_k.new_zeros([1]), seq_lens_k
                                ]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
        max_seqlen_q=int(seq_lens_q.max()),
        max_seqlen_k=int(seq_lens_k.max()))

    # Unpack: restore to padded batch format with zeros in padding positions
    x = unpack_tensor(x_packed, mask_q, (b, seq_len_padded, num_heads, head_dim))
    return x.type(out_dtype)


# ----------------------------------------------------------------------------
# PYTORCH SDPA FALLBACK IMPLEMENTATION
# ----------------------------------------------------------------------------
def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_q: torch.Tensor,
    mask_k: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch SDPA fallback for attention with variable sequence lengths.

    Uses PyTorch's scaled_dot_product_attention with attention masking
    to handle variable-length sequences. When all positions are valid
    (no padding), skips masking entirely to allow SDPA to use its most
    efficient kernel (FlashAttention-2).

    Args:
        q (torch.Tensor):
            Query tensor of shape [B, L1, num_heads, head_dim]
        k (torch.Tensor):
            Key tensor of shape [B, L2, num_heads, head_dim]
        v (torch.Tensor):
            Value tensor of shape [B, L2, num_heads, head_dim]
        mask_q (torch.Tensor):
            Boolean mask for valid query positions, shape [B, L1].
            True indicates valid positions.
        mask_k (torch.Tensor):
            Boolean mask for valid key/value positions, shape [B, L2].
            True indicates valid positions.

    Returns:
        torch.Tensor:
            Output tensor of shape [B, L1, num_heads, head_dim]
    """
    out_dtype = q.dtype

    # SDPA expects [B, num_heads, L, head_dim]
    q = q.transpose(1, 2)  # [B, num_heads, L1, head_dim]
    k = k.transpose(1, 2)  # [B, num_heads, L2, head_dim]
    v = v.transpose(1, 2)  # [B, num_heads, L2, head_dim]

    # Check if all positions are valid (no padding)
    # When true, we can skip masking and let SDPA use its fastest kernel
    all_valid = mask_q.all() and mask_k.all()

    if all_valid:
        # No masking needed - SDPA can use FlashAttention-2 kernel
        x = F.scaled_dot_product_attention(q, k, v)
    else:
        # Build attention mask: valid positions get 0.0, invalid get -inf
        # Combined mask shape: [B, L1, L2] -> [B, 1, L1, L2] for head broadcasting
        attn_mask = mask_q[:, :, None] & mask_k[:, None, :]  # [B, L1, L2]
        attn_mask = attn_mask.unsqueeze(1)  # [B, 1, L1, L2]

        # Convert boolean mask to float: True->0.0, False->-inf
        attn_mask = torch.where(attn_mask, 0.0, float('-inf'))
        attn_mask = attn_mask.to(q.dtype)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    # Transpose back to [B, L1, num_heads, head_dim]
    x = x.transpose(1, 2)

    # Zero out padding positions in output (only needed when there's padding)
    if not all_valid:
        x = x * mask_q[:, :, None, None]

    return x.type(out_dtype)


# ----------------------------------------------------------------------------
# PACKING/UNPACKING HELPERS
# ----------------------------------------------------------------------------
# These operations use dynamic boolean indexing which is not supported by
# torch.compile, hence the @torch.compiler.disable decorator.

@torch.compiler.disable
def pack_tensor(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Pack a batched tensor by selecting only valid (non-padding) positions.

    Args:
        tensor (torch.Tensor):
            Input tensor of shape [B, L, ...]
        mask (torch.Tensor):
            Boolean mask of shape [B, L] where True = valid position

    Returns:
        torch.Tensor:
            Packed tensor of shape [total_valid_positions, ...] containing
            only the elements at valid positions
    """
    return tensor[mask]


@torch.compiler.disable
def unpack_tensor(packed: torch.Tensor, mask: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Unpack a tensor by placing values back at their original positions.

    Args:
        packed (torch.Tensor):
            Packed tensor of shape [total_valid_positions, ...]
        mask (torch.Tensor):
            Boolean mask of shape [B, L] indicating where to place values
        shape (tuple):
            Target output shape [B, L, ...]

    Returns:
        torch.Tensor:
            Unpacked tensor with zeros in padding positions
    """
    unpacked = torch.zeros(shape, dtype=packed.dtype, device=packed.device)
    unpacked[mask] = packed
    return unpacked
