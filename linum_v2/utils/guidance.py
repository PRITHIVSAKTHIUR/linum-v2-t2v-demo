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
File: guidance.py
Description: Adaptive Projected Guidance (APG) components for diffusion/flow-matching models.

APG improves upon standard Classifier-Free Guidance by decomposing the guidance
update into parallel and perpendicular components relative to the conditional
prediction, reducing oversaturation at high guidance scales while maintaining
quality improvements.

Reference:
    Sadat et al., "Eliminating Oversaturation and Artifacts of High Guidance
    Scales in Diffusion Models", ICLR 2025.
    https://arxiv.org/abs/2410.02416
"""
import torch
from typing import Optional, Tuple


# ----------------------------------------------------------------------------
# APG (Adaptive Projected Guidance) Components
# ----------------------------------------------------------------------------
class MomentumBuffer:
    """Momentum buffer for APG to track exponential moving average of updates."""

    def __init__(self, momentum: float):
        """
        Initialize momentum buffer.

        Args:
            momentum (float): Momentum coefficient for exponential moving average.
        """
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor) -> None:
        """
        Update the running average with a new value.

        Args:
            update_value (torch.Tensor): The new update value to incorporate.
        """
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(v0: torch.Tensor, v1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project v0 onto v1 to get parallel and orthogonal components.

    Args:
        v0 (torch.Tensor): Vector to project (B, C, T, H, W) or (B, C, H, W).
        v1 (torch.Tensor): Vector to project onto (B, C, T, H, W) or (B, C, H, W).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Parallel and orthogonal components of v0.
    """
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=list(range(1, v1.ndim)))  # Normalize v1
    v0_parallel = (v0 * v1).sum(dim=list(range(1, v1.ndim)), keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


# ----------------------------------------------------------------------------
# APG (Adaptive Projected Guidance) Calculation
# ----------------------------------------------------------------------------
def adaptive_projected_guidance(
    pred_cond: torch.Tensor,
    pred_other: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: Optional[MomentumBuffer] = None,
    eta: float = 0.0,
    rescale: float = 20.0,
    spatial_rescale_only: bool = True,
) -> torch.Tensor:
    """
    Apply Adaptive Projected Guidance (APG) to diffusion/flow-matching model predictions.

    APG decomposes the guidance update into parallel and perpendicular components
    relative to the conditional prediction, reducing oversaturation at high guidance
    scales while maintaining quality improvements.

    Args:
        pred_cond (torch.Tensor): Conditional model prediction (B, C, T, H, W) or (B, C, H, W).
        pred_uncond (torch.Tensor): Unconditional model prediction (B, C, T, H, W) or (B, C, H, W).
        guidance_scale (float): CFG scale factor.
        momentum_buffer (Optional[MomentumBuffer]): Buffer for momentum-based updates.
        eta (float): Scale factor for parallel component (default 0.0).
        rescale (float): Rescaling factor for adaptive normalization (default 20.0).
        spatial_rescale_only (bool): For 5D video tensors, rescale only spatially (C, H, W)
            instead of across all dimensions (C, T, H, W). Default True, as this preserves
            temporal dynamics and motion coherence while still controlling spatial magnitude.

    Returns:
        torch.Tensor: Guided prediction with APG applied.
    """
    diff = pred_cond - pred_other

    # Apply momentum if buffer is provided
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    # Apply rescaling if rescale factor is set
    if rescale > 0:
        ones = torch.ones_like(diff)

        # Compute norm based on tensor dimensions and spatial_rescale_only flag
        if spatial_rescale_only and diff.ndim == 5:
            # For video tensors (B, C, T, H, W), compute norm per frame.
            # This preserves temporal dynamics and motion coherence - rescaling across
            # all frames would dampen motion and cause temporal smoothing artifacts.
            diff_norm = diff.norm(p=2, dim=[1, 3, 4], keepdim=True)  # (B, 1, T, 1, 1)
        else:
            # Default behavior: norm across all non-batch dimensions
            diff_norm = diff.norm(p=2, dim=list(range(1, diff.ndim)), keepdim=True)

        scale_factor = torch.minimum(ones, rescale / diff_norm)
        diff = diff * scale_factor

    # Project difference into parallel and orthogonal components
    diff_parallel, diff_orthogonal = project(diff, pred_cond)

    # Combine components with eta weighting
    normalized_update = diff_orthogonal + eta * diff_parallel

    # Apply guidance
    pred_guided = pred_cond + guidance_scale * normalized_update

    return pred_guided
