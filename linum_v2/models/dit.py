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
File: dit.py
Description: Implements the DiT (Diffusion Transformer) video model for text-to-video generation.
"""
import math
from typing import List, Tuple, Optional

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from linum_v2.models.attention import compute_attention

torch._dynamo.config.capture_scalar_outputs = True


# ----------------------------------------------------------------------------
# DIT MODEL
# ----------------------------------------------------------------------------
class DiTModel(ModelMixin, ConfigMixin):
    """
    Diffusion transformer backbone that can be used for text-to-video (T2V) generation.

    It uses a patchified 3D embedding for videos, modulated transformer blocks with self-attention
    and cross attention, and rotary positional embeddings.
    """

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['DiTAttentionBlock']

    @register_to_config
    def __init__(self,
                 patch_size: Tuple[int, int, int],
                 text_len: int,
                 in_dim: int,
                 dim: int,
                 ffn_dim: int,
                 freq_dim: int,
                 text_dim: int,
                 out_dim: int,
                 num_heads: int,
                 num_layers: int,
                 qk_norm: bool,
                 cross_attn_norm: bool,
                 zero_init_head_and_bias: bool,
                 epsilon: float,
                 layernorm_elementwise_affine: bool = True):
        """
        Initialize the diffusion model backbone.

        Args:
            patch_size (Tuple[int, int, int]):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
            text_len (int):
                Fixed length for text embeddings.
            in_dim (int):
                Input video channels (C_in).
            dim (int):
                Hidden dimension of the transformer.
            ffn_dim (int):
                Intermediate dimension in feed-forward network.
            freq_dim (int):
                Dimension for sinusoidal time embeddings.
            text_dim (int):
                Input dimension for text embeddings.
            out_dim (int):
                Output video channels (C_out).
            num_heads (int):
                Number of attention heads.
            num_layers (int):
                Number of transformer blocks.
            qk_norm (bool):
                Enable query/key normalization.
            cross_attn_norm (bool):
                Enable cross-attention normalization.
            zero_init_head_and_bias (bool):
                Whether to initialize the head and bias to zero.
            epsilon (float):
                Epsilon value for normalization layers.
            layernorm_elementwise_affine (bool, optional):
                Whether LayerNorm layers should have learnable affine parameters.
                Default: True
        """

        super().__init__()

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.epsilon = epsilon

        # Embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(),
                                            nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Text conditioning layers must use bias=False. With bias enabled, padded tokens
        # (which are zeros) would be summed with the bias term, causing information to
        # leak into padding positions. This breaks the assumption that padding is ignored.
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim, bias=False),
            nn.GELU(approximate='tanh'), nn.Linear(dim, dim, bias=False))

        # Blocks (work-horse of the model)
        self.blocks = nn.ModuleList([
            DiTAttentionBlock(dim=dim,
                              ffn_dim=ffn_dim,
                              num_heads=num_heads,
                              qk_norm=qk_norm,
                              cross_attn_norm=cross_attn_norm,
                              epsilon=epsilon,
                              layernorm_elementwise_affine=layernorm_elementwise_affine)
            for _ in range(num_layers)
        ])

        # Head (output layer)
        self.head = OutputHead(dim=dim,
                         out_dim=out_dim,
                         patch_size=patch_size,
                         epsilon=epsilon,
                         layernorm_elementwise_affine=layernorm_elementwise_affine)

        # Precompute rotary position embedding frequencies
        # Not using register_buffer to avoid dtype changes during to() calls
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        head_dim = dim // num_heads
        self.freqs = torch.cat([
            compute_rotary_frequencies(1024, head_dim - 4 * (head_dim // 6)),  # temporal
            compute_rotary_frequencies(1024, 2 * (head_dim // 6)),  # height
            compute_rotary_frequencies(1024, 2 * (head_dim // 6))   # width
        ], dim=1)

        # Initialize weights
        self.init_weights(zero_init_head_and_bias=zero_init_head_and_bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_input: List[torch.Tensor],
        frame_lens: torch.Tensor,
        text_input_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass through the DiT.

        Processes input video tensors through the transformer backbone with
        conditioning from text/image embeddings and timestep embeddings.
        Assumes every sample in the batch x has the same dimensions (c, t, h, w).
        Padding for patch divisibility is handled internally.

        Args:
            x (torch.Tensor):
                Input video tensor of shape (B C F H W)
            t (torch.Tensor):
                Timesteps tensor of shape [B]
            text_input (List[torch.Tensor]):
                List of text embeddings, each with shape [L, C_text_emb]
            frame_lens (torch.Tensor):
                Actual frame counts before padding, shape [B]. Always provided (1 for images).
            text_input_lens (Optional[torch.Tensor]):
                Lengths of the text embeddings, shape [B]. Required during training.
        Returns:
            torch.Tensor:
                Denoised video tensor of shape [B, C_out, F_orig, W_orig],
                cropped to original input spatial dimensions if padding occurred.
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Pad input if H or W are not divisible by patch size
        x_padded, h_orig, w_orig = self._pad_input_if_needed(
            x=x, original_hw=(x.shape[3], x.shape[4]))  # (B C F H_pad W_pad)

        # embeddings: Apply patch embedding to the batched padded tensor
        # Input: (B C_in F H_pad W_pad)
        # Output: (B dim F//pt H_pad//ph W_pad//pw)
        x_embedded = self.patch_embedding(x_padded)

        # Patch grid dimensions after embedding: (F_p, H_p, W_p)
        # Since all samples are assumed to have same dimensions, this is a single set of dimensions
        # Extract patch dimensions F_p, H_p, W_p
        # These are int or torch.SymInt when compiled
        f_p = x_embedded.shape[2]
        h_p = x_embedded.shape[3]
        w_p = x_embedded.shape[4]

        # Flatten spatial-temporal dims and transpose for transformer
        # Input: (B dim F_p H_p W_p) -> Output: (B F_p*H_p*W_p dim)
        x_flat = x_embedded.flatten(2).transpose(1, 2)  # Shape (B L_padded dim)

        # Get actual sequence lengths. Since we assume no padding to max_seq_len within batch yet,
        # and all samples have same dim, all seq_lens are L_padded.
        # Shape (B)
        b, seq_len_padded, _ = x_flat.shape

        # Create frame mask
        # Each frame contributes h_p * w_p positions to the sequence
        seq_lens = frame_lens * h_p * w_p

        # Create position indices for each sequence in the batch
        position_ids = torch.arange(seq_len_padded, device=x_flat.device).unsqueeze(0).expand(b, -1)

        # Create mask: True for valid positions, False for padding
        frame_mask = position_ids < seq_lens.unsqueeze(1)  # [B, L] - Keep as 2D

        # Apply mask to x_flat to zero out padded positions (including patch embedding bias)
        x_flat = x_flat.masked_fill(~frame_mask.unsqueeze(-1), 0.0)  # Use masked_fill

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(dim=self.freq_dim, position=t, dtype=x_flat.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # text embeddings
        if self.training:
            assert text_input_lens is not None, "text_input_lens must be provided when training"
        text_embedding = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in text_input
            ]))

        # arguments
        kwargs = dict(e=e0,
                      f_p=f_p,
                      h_p=h_p,
                      w_p=w_p,
                      freqs=self.freqs,
                      text_embedding=text_embedding,
                      text_embedding_lens=text_input_lens,
                      frame_mask=frame_mask)

        x_block = x_flat
        for block in self.blocks:
            x_block = block(x_block, **kwargs)

        # head
        x_head = self.head(x_block, e=e, frame_mask=frame_mask)

        # unpatchify
        return self.unpatchify(x_head=x_head,
                               f_p=f_p,
                               h_p=h_p,
                               w_p=w_p,
                               original_hw_tuple=(h_orig, w_orig),
                               seq_len=seq_len_padded)

    def _pad_input_if_needed(self, x: torch.Tensor,
                             original_hw: Tuple[int, int]) -> Tuple[torch.Tensor, int, int]:
        """
        Pads the input tensor's spatial dimensions if they are not divisible by patch sizes.

        Args:
            x (torch.Tensor):
                Input video tensor of shape (B C F H W)
            original_hw (Tuple[int, int]):
                Tuple containing the original (H_orig, W_orig) of the input.

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]:
                A tuple containing:
                - The potentially padded video tensor.
                - The original_hw tuple (passed through).
        """
        H_orig, W_orig = original_hw
        _, ph, pw = self.patch_size  # t_patch, h_patch, w_patch

        pad_h = (ph - H_orig % ph) % ph
        pad_w = (pw - W_orig % pw) % pw

        if pad_h > 0 or pad_w > 0:
            # Pad format for 5D tensor (B C F H W) for F.pad is:
            # (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
            # We only pad right and bottom.
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return x, H_orig, W_orig

    def unpatchify(self, x_head: torch.Tensor, f_p: int, h_p: int, w_p: int,
                   original_hw_tuple: Tuple[int,
                                            int], seq_len: int) -> torch.Tensor:
        """
        Reconstruct latent tensor from patch token sequence.

        This reverses the patchification done by self.patch_embedding. It does NOT
        perform VAE decoding - the output is still in latent space.

        Pipeline context:
            [VAE Latent] -> patch_embedding -> [Patch Tokens] -> transformer
                -> [Patch Tokens] -> unpatchify -> [VAE Latent]

        Args:
            x_head (torch.Tensor):
                Output from the model head containing predicted patch values,
                shape [B, num_patches, C_out * prod(patch_size)].
            f_p (int):
                Number of patches in the frame/temporal dimension.
            h_p (int):
                Number of patches in the height dimension.
            w_p (int):
                Number of patches in the width dimension.
            original_hw_tuple (Tuple[int, int]):
                Original (H, W) of the latent before any padding for patch divisibility.
            seq_len (int):
                Total number of patches (should equal f_p * h_p * w_p).

        Returns:
            torch.Tensor:
                Reconstructed latent tensor of shape [B, C_out, F, H_orig, W_orig],
                cropped to remove any padding that was added for patch divisibility.
        """
        B = x_head.shape[0]  # B is an int or SymInt
        c_out = self.out_dim  # C_out
        pt, ph, pw = self.patch_size  # t_patch, h_patch, w_patch

        # f_p, h_p, w_p are already SymInts if dynamic
        num_patches = f_p * h_p * w_p  # SymInt operation
        # Ensure seq_len is also treated as SymInt for comparison
        seq_len_from_input = x_head.shape[1]  # This is the most reliable SymInt for seq_len here

        assert num_patches == seq_len_from_input, "Calculated num_patches must equal sequence length from x_head"
        assert seq_len_from_input == seq_len, "x_head sequence dim must equal common_seq_len_val passed as argument"

        # Reshape the flattened patches
        # Input: [B, common_seq_len, C_out * pt * ph * pw]
        # Output: [B, F_p, H_p, W_p, pt, ph, pw, C_out]
        u_reshaped = x_head.view(B, f_p, h_p, w_p, pt, ph, pw, c_out)

        # Permute dimensions using einsum: 'bfhwpqrc->bcfphqwr'
        # Output: [B, C_out, F_p, pt, H_p, ph, W_p, pw]
        u_permuted = torch.einsum('bfhwpqrc->bcfphqwr', u_reshaped)

        # Reshape to combine patch dimensions, resulting in padded shape
        # Output: [B, C_out, F_p*pt, H_p*ph, W_p*pw]
        f_padded_full = f_p * pt
        h_padded_full = h_p * ph
        w_padded_full = w_p * pw
        u_reconstructed = u_permuted.reshape(B, c_out, f_padded_full, h_padded_full, w_padded_full)

        # Crop back to original H and W dimensions
        h_orig, w_orig = original_hw_tuple
        u_cropped = u_reconstructed[..., :h_orig, :w_orig]

        return u_cropped

    def init_weights(self, zero_init_head_and_bias: bool = True):
        """
        Initialize model parameters using Xavier and normal distributions.

        This method applies Xavier uniform initialization to linear layers, normal
        initialization to embedding layers, and zeros initialization to the output layer.
        """
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None and zero_init_head_and_bias:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        if zero_init_head_and_bias:
            nn.init.zeros_(self.head.head.weight)

    def count_parameters(self, only_trainable: bool = True) -> int:
        """
        Count the number of parameters in the model.

        Args:
            only_trainable (bool, optional): 
                If True, only count trainable parameters. Default: True

        Returns:
            int: Total number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# DIT SELF ATTENTION & CROSS ATTENTION BLOCKS
# ----------------------------------------------------------------------------
class DiTSelfAttention(nn.Module):
    """
    Self-attention module for the DiT model.

    This class implements multi-head self-attention with rotary positional embeddings
    and optional query/key normalization.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qk_norm: bool = True,
                 epsilon: float = 1e-6):
        """
        Initialize the self-attention module.

        Args:
            dim (int):
                Hidden dimension of the attention module
            num_heads (int):
                Number of attention heads
            qk_norm (bool, optional):
                Whether to apply normalization to query and key. Default: True
            epsilon (float, optional):
                Epsilon value for normalization. Default: 1e-6
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.epsilon = epsilon

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = DiTRMSNorm(dim, epsilon=epsilon) if qk_norm else nn.Identity()
        self.norm_k = DiTRMSNorm(dim, epsilon=epsilon) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, f_p: int, h_p: int, w_p: int, freqs: torch.Tensor,
                frame_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies self-attention to the input using rotary positional embeddings.

        Args:
            x (torch.Tensor):
                Input tensor of shape [B, L, C] where B is batch size, L is sequence length,
                and C is the channel dimension. L must be f_p * h_p * w_p.
            f_p (int): Number of patches in the temporal dimension.
            h_p (int): Number of patches in the height dimension.
            w_p (int): Number of patches in the width dimension.
            freqs (torch.Tensor):
                Precomputed frequencies for rotary embeddings.
            frame_mask (torch.Tensor):
                Pre-computed mask for valid positions, shape [B, L]. Always provided.

        Returns:
            torch.Tensor:
                Output tensor after self-attention of shape [B, L, C]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # s (sequence length) must be equal to f_p * h_p * w_p
        # This assertion can be helpful for debugging, especially with SymInts
        # assert s == f_p * h_p * w_p, f"Sequence length {s} does not match f_p*h_p*w_p {f_p*h_p*w_p}"

        def qkv_fn(x: torch.Tensor,
                   mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Apply linear projections to the input tensor and apply mask to the projections.
            """
            # Apply linear projections
            q_proj = self.q(x)
            k_proj = self.k(x)
            v_proj = self.v(x)

            # Mask after projection to zero out bias effects on padded positions
            q_proj = q_proj.masked_fill(~mask.unsqueeze(-1), 0.0)
            k_proj = k_proj.masked_fill(~mask.unsqueeze(-1), 0.0)
            v_proj = v_proj.masked_fill(~mask.unsqueeze(-1), 0.0)

            # Apply normalization and reshape
            q = self.norm_q(q_proj).view(b, s, n, d)
            k = self.norm_k(k_proj).view(b, s, n, d)
            v = v_proj.view(b, s, n, d)
            return q, k, v

        # q, k, v shapes: [B, L, num_heads, head_dim] where L = f_p * h_p * w_p
        q, k, v = qkv_fn(x=x, mask=frame_mask)

        # Compute attention - pass masks directly
        x = compute_attention(q=apply_rope_embeddings(x=q, f_s=f_p, h_s=h_p, w_s=w_p, freqs=freqs),
                            k=apply_rope_embeddings(x=k, f_s=f_p, h_s=h_p, w_s=w_p, freqs=freqs),
                            v=v,
                            mask_q=frame_mask,
                            mask_k=frame_mask)  # [B, L, num_heads, head_dim]

        x = x.flatten(2)  # [B, L, num_heads * head_dim] -> [B, L, C]

        # Mask output to ensure padded positions remain zero after bias
        x = self.o(x)  # [B, L, C]
        x = x.masked_fill(~frame_mask.unsqueeze(-1), 0.0)
        return x


class DiTCrossAttention(nn.Module):
    """
    Cross-attention module for text-to-video generation in the DiT model.

    This class implements multi-head cross-attention between video features and text features.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qk_norm: bool = True,
                 epsilon: float = 1e-6):
        """
        Initialize the cross-attention module.

        Args:
            dim (int):
                Hidden dimension of the attention module
            num_heads (int):
                Number of attention heads
            qk_norm (bool, optional):
                Whether to apply normalization to query and key. Default: True
            epsilon (float, optional):
                Epsilon value for normalization. Default: 1e-6
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.epsilon = epsilon

        # layers
        # The query is the image/video features. K/V are the text features.
        # Key and value projections for text features must use bias=False. With bias
        # enabled, padded tokens (zeros) would be summed with the bias, causing
        # information to leak into padding positions.
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.norm_q = DiTRMSNorm(dim, epsilon=epsilon) if qk_norm else nn.Identity()
        self.norm_k = DiTRMSNorm(dim, epsilon=epsilon) if qk_norm else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            text_embedding: torch.Tensor,
            text_embedding_lens: Optional[torch.Tensor],
            frame_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies cross-attention between video features and text features.

        Args:
            x (torch.Tensor):
                Video features tensor of shape [B, L1, C]
            text_embedding (torch.Tensor):
                Text features tensor of shape [B, L2, C]
            text_embedding_lens (Optional[torch.Tensor]):
                Lengths of text sequences of shape [B], can be None during inference
            frame_mask (torch.Tensor):
                Pre-computed mask for valid positions, shape [B, L]. Always provided.

        Returns:
            torch.Tensor:
                Output tensor after cross-attention of shape [B, L1, C]
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # Compute query from video features
        # Mask query projection to zero out bias effects on padded positions
        q_proj = self.q(x)
        q_proj = q_proj.masked_fill(~frame_mask.unsqueeze(-1), 0.0)  # Always apply
        q = self.norm_q(q_proj).view(b, -1, n, d)  # [B, L1, num_heads, head_dim]

        # Compute key, value from text features (no masking needed)
        k = self.norm_k(self.k(text_embedding)).view(b, -1, n, d)  # [B, L2, num_heads, head_dim]
        v = self.v(text_embedding).view(b, -1, n, d)  # [B, L2, num_heads, head_dim]

        # Create mask_k from text_embedding_lens if provided
        if text_embedding_lens is not None:
            # Create boolean mask for text features
            l2 = text_embedding.shape[1]
            arange_l2 = torch.arange(l2, device=text_embedding.device)
            mask_k = arange_l2[None, :] < text_embedding_lens[:, None]  # [B, L2]
        else:
            # If no text_embedding_lens, assume all positions are valid
            mask_k = torch.ones(b, text_embedding.shape[1], dtype=torch.bool, device=text_embedding.device)

        # Compute attention with appropriate masking
        x = compute_attention(q=q, k=k, v=v, mask_q=frame_mask,
                            mask_k=mask_k)  # [B, L1, num_heads, head_dim]

        # Output projection
        # Mask output to ensure padded positions remain zero after bias
        x = x.flatten(2)  # [B, L1, num_heads * head_dim] -> [B, L1, C]
        x = self.o(x)  # [B, L1, C]
        x = x.masked_fill(~frame_mask.unsqueeze(-1), 0.0)  # Always apply
        return x


# ----------------------------------------------------------------------------
# DIT ATTENTION BLOCK
# ----------------------------------------------------------------------------
class DiTAttentionBlock(nn.Module):
    """
    Transformer block used in the DiT model with self-attention and cross-attention.

    This class implements a transformer block with self-attention followed by
    cross-attention and a feed-forward network.

    Each component includes modulation via learned parameters and normalization.
    """

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 qk_norm: bool = True,
                 cross_attn_norm: bool = False,
                 epsilon: float = 1e-6,
                 layernorm_elementwise_affine: bool = True):
        """
        Initialize the transformer block.

        Args:
            dim (int):
                Hidden dimension of the transformer block
            ffn_dim (int):
                Intermediate dimension in feed-forward network
            num_heads (int):
                Number of attention heads
            qk_norm (bool, optional):
                Whether to apply normalization to query and key. Default: True
            cross_attn_norm (bool, optional):
                Whether to use normalization before cross-attention. Default: False
            epsilon (float, optional):
                Epsilon value for normalization. Default: 1e-6
            layernorm_elementwise_affine (bool, optional):
                Whether LayerNorm layers should have learnable affine parameters.
                Default: True
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.epsilon = epsilon

        # layers
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=layernorm_elementwise_affine)
        self.self_attn = DiTSelfAttention(dim=dim,
                                          num_heads=num_heads,
                                          qk_norm=qk_norm,
                                          epsilon=epsilon)
        self.norm3 = nn.LayerNorm(dim,
                                  elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = DiTCrossAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            epsilon=epsilon)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=layernorm_elementwise_affine)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
                                 nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        f_p: int,
        h_p: int,
        w_p: int,
        freqs: torch.Tensor,
        text_embedding: torch.Tensor,
        text_embedding_lens: Optional[torch.Tensor],
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x (torch.Tensor):
                Input tensor of shape [B, L, C]. L must be f_p * h_p * w_p.
            e (torch.Tensor):
                Time embedding tensor of shape [B, 6, C]
            f_p (int): Number of patches in the temporal dimension.
            h_p (int): Number of patches in the height dimension.
            w_p (int): Number of patches in the width dimension.
            freqs (torch.Tensor):
                Precomputed frequencies for rotary embeddings.
            text_embedding (torch.Tensor):
                Text embedding tensor of shape [B, L2, C]
            text_embedding_lens (Optional[torch.Tensor]):
                Lengths of text embedding sequences of shape [B], can be None during inference
            frame_mask (torch.Tensor):
                Pre-computed mask for valid positions, shape [B, L]. Always provided.

        Returns:
            torch.Tensor:
                Output tensor after transformer block of shape [B, L, C]
        """
        # ------------------------------------------------------------
        # Normalize + Modulate Input
        # ------------------------------------------------------------
        e = (self.modulation + e).chunk(6, dim=1)
        norm_modulated_x = self.norm1(x) * (1 + e[1]) + e[0]

        # Self Attention
        y = self.self_attn(x=norm_modulated_x,
                           f_p=f_p,
                           h_p=h_p,
                           w_p=w_p,
                           freqs=freqs,
                           frame_mask=frame_mask)

        # Skip connection w/ Modulated Self-attention
        x = x + y * e[2]

        # Cross Attention + Skip connection
        x = x + self.cross_attn(
            x=self.norm3(x), text_embedding=text_embedding, text_embedding_lens=text_embedding_lens,
            frame_mask=frame_mask)

        # Normalization -> Modulation -> FFN
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])  # [B, L, C]

        # Mask FFN output to prevent bias leakage
        # Expand mask from [B, L] to [B, L, 1] for broadcasting
        y = y * frame_mask.unsqueeze(-1)

        # Modulation -> Skip Connection
        x = x + y * e[5]
        return x


# ----------------------------------------------------------------------------
# DIT NORMALIZATION
# ----------------------------------------------------------------------------
class DiTRMSNorm(nn.Module):
    """
    RMS normalization layer for the DiT model.

    This normalization layer normalizes inputs by dividing by the root mean square
    of the input values, with a learnable scale parameter.
    """

    def __init__(self, dim: int, epsilon: float = 1e-5):
        """
        Initialize the RMS normalization layer.

        Args:
            dim (int):
                Dimension to normalize over
            epsilon (float, optional):
                Small constant for numerical stability. Default: 1e-5
        """
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input tensor.

        Args:
            x (torch.Tensor):
                Input tensor of shape [B, L, C]

        Returns:
            torch.Tensor:
                Normalized tensor of shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal method to compute root mean square normalization.

        Args:
            x (torch.Tensor):
                Input tensor

        Returns:
            torch.Tensor:
                Normalized tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)


# ----------------------------------------------------------------------------
# DIT Output HEAD
# ----------------------------------------------------------------------------
class OutputHead(nn.Module):
    """
    Output head for the DiT model.

    This module applies normalization, modulation, and a final linear projection
    to convert hidden representations to the output space.
    """

    def __init__(self,
                 dim: int,
                 out_dim: int,
                 patch_size: Tuple[int, int, int],
                 epsilon: float = 1e-6,
                 layernorm_elementwise_affine: bool = True):
        """
        Initialize the output head.

        Args:
            dim (int):
                Input hidden dimension
            out_dim (int):
                Output channels dimension
            patch_size (Tuple[int, int, int]):
                3D patch dimensions (time, height, width)
            epsilon (float, optional):
                Small constant for normalization. Default: 1e-6
            layernorm_elementwise_affine (bool, optional):
                Whether LayerNorm layers should have learnable affine parameters.
                Default: True
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.epsilon = epsilon

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=layernorm_elementwise_affine)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply the output head to transform features to output space.

        Args:
            x (torch.Tensor):
                Input feature tensor of shape [B, L, C]
            e (torch.Tensor):
                Time embedding tensor of shape [B, C]
            frame_mask (torch.Tensor):
                Pre-computed mask for valid positions, shape [B, L]. Always provided.

        Returns:
            torch.Tensor:
                Output tensor of shape [B, L, C_out * prod(patch_size)]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))

        # Mask output to ensure padded positions remain zero after bias
        # Expand mask from [B, L] to [B, L, 1] for broadcasting
        x = x * frame_mask.unsqueeze(-1)

        return x


# ----------------------------------------------------------------------------
# EMBEDDING FUNCTIONS
# ----------------------------------------------------------------------------
def sinusoidal_embedding_1d(dim: int,
                            position: torch.Tensor,
                            dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Generate sinusoidal embeddings for timestep encoding.

    Creates sinusoidal embeddings based on positions for time encoding in the diffusion process.

    Args:
        dim (int):
            Dimension of the output embeddings, must be even
        position (torch.Tensor):
            Position tensor of shape [B] containing timesteps

    Returns:
        torch.Tensor:
            Sinusoidal embeddings of shape [B, dim]
    """
    assert dim % 2 == 0
    embedding_dim_half = dim // 2
    positions_float = position.to(torch.float64)

    # Compute inverse frequencies: 1/10000^(2i/d) for dimension indices i
    inv_freq = torch.pow(10000.0, -torch.arange(embedding_dim_half).to(positions_float) / embedding_dim_half)
    angle_rads = torch.outer(positions_float, inv_freq)

    # Concatenate cos and sin components
    pos_encoding = torch.cat([torch.cos(angle_rads), torch.sin(angle_rads)], dim=1)
    return pos_encoding.to(dtype)


@amp.autocast(enabled=False, device_type='cuda')
def compute_rotary_frequencies(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """
    Generate complex-valued rotary position embedding frequencies.

    Creates unit complex numbers representing rotations for RoPE (Rotary Position Embedding).
    Each position gets a unique rotation in the complex plane based on its index.

    Args:
        max_seq_len (int):
            Maximum sequence length
        dim (int):
            Dimension of the positional embeddings, must be even
        theta (float, optional):
            Base for the frequency calculation. Default: 10000

    Returns:
        torch.Tensor:
            Complex-valued tensor of shape [max_seq_len, dim/2] containing unit rotations
    """
    assert dim % 2 == 0
    seq_positions = torch.arange(max_seq_len, dtype=torch.float64)
    dim_indices = torch.arange(0, dim, 2, dtype=torch.float64)
    inv_freq_base = 1.0 / torch.pow(theta, dim_indices / dim)

    # Outer product gives rotation angle for each (position, dimension) pair
    rotation_angles = torch.outer(seq_positions, inv_freq_base)

    # Convert angles to unit complex numbers (cos + i*sin)
    complex_rotations = torch.polar(torch.ones_like(rotation_angles), rotation_angles)
    return complex_rotations


@amp.autocast(enabled=False, device_type='cuda')
def apply_rope_embeddings(x: torch.Tensor, f_s: int, h_s: int, w_s: int, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to the input tensor.

    This version assumes all samples in a batch share the same patch dimensions (f_s, h_s, w_s).
    These dimensions can vary between batches, and torch.compile handles this by creating
    specializations based on the sequence length derived from f_s * h_s * w_s.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, S, num_heads, head_dim].
            S is assumed to be common_seq_len.
        f_s (int): # SymInt
            Grid dimension F (frames/time) common to all samples.
        h_s (int): # SymInt
            Grid dimension H (height) common to all samples.
        w_s (int): # SymInt
            Grid dimension W (width) common to all samples.
        freqs (torch.Tensor):
            Precomputed rotary embedding frequencies of shape [max_rope_len, head_dim/2].
            (Where head_dim/2 is the complex dimension for frequencies).

    Returns:
        torch.Tensor:
            Tensor with rotary positional embeddings applied, shape [B, S, num_heads, head_dim].
    """
    b, _, n, d_head = x.shape

    # f_s, h_s, w_s are already SymInts if shapes are dynamic
    common_seq_len = f_s * h_s * w_s  # This will be a SymInt or int
    head_dim_half = d_head // 2  # Dimension for complex numbers (freqs are defined over this)

    # Split frequencies into temporal, height, and width components
    # RoPE encodes 3D position (frame, height, width) using separate frequency bands
    temporal_freq_dim = head_dim_half - 2 * (head_dim_half // 3)
    spatial_freq_dim = head_dim_half // 3
    freq_temporal, freq_height, freq_width = freqs.split(
        [temporal_freq_dim, spatial_freq_dim, spatial_freq_dim], dim=1)

    # Build 3D frequency grid by broadcasting each component along its axis
    # Shape for each: [f_s, h_s, w_s, freq_dim_for_that_axis]
    freq_t_grid = freq_temporal.narrow(0, 0, f_s).view(f_s, 1, 1, -1).expand(f_s, h_s, w_s, -1)
    freq_h_grid = freq_height.narrow(0, 0, h_s).view(1, h_s, 1, -1).expand(f_s, h_s, w_s, -1)
    freq_w_grid = freq_width.narrow(0, 0, w_s).view(1, 1, w_s, -1).expand(f_s, h_s, w_s, -1)

    # Concatenate along the last dimension to get [f_s, h_s, w_s, head_dim_half]
    freqs_volume = torch.cat([freq_t_grid, freq_h_grid, freq_w_grid], dim=-1)

    # Reshape to [common_seq_len, head_dim_half] and add unsqueeze for num_heads broadcasting
    # Final shape for broadcasting: [common_seq_len, 1, head_dim_half]
    freqs_for_broadcast = freqs_volume.reshape(common_seq_len, head_dim_half).unsqueeze(1)

    # Prepare input tensor x for complex multiplication
    # x original shape: [B, common_seq_len, N, D_head]
    # Reshape to [B, common_seq_len, N, D_head_half, 2] for view_as_complex
    x_reshaped_for_complex = x.to(torch.float64).reshape(b, common_seq_len, n, head_dim_half, 2)
    x_complex = torch.view_as_complex(x_reshaped_for_complex)
    # x_complex shape: [B, common_seq_len, N, D_head_half]

    # Apply rotary embedding (element-wise multiplication, broadcasting freqs)
    # x_complex:        [B, common_seq_len, N, D_head_half]
    # freqs_for_broadcast: [1, common_seq_len, 1, D_head_half] (after implicit broadcasting for batch dim)
    # Result:           [B, common_seq_len, N, D_head_half]
    x_rotated_complex = x_complex * freqs_for_broadcast.unsqueeze(0)  # Add batch dim to freqs

    # Convert back to real representation
    # Output shape: [B, common_seq_len, N, D_head_half, 2]
    x_rotated_real_pairs = torch.view_as_real(x_rotated_complex)

    # Flatten the last two dimensions to get back to [B, common_seq_len, N, D_head]
    output = x_rotated_real_pairs.flatten(3)

    return output.to(x.dtype)
