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
File: latents.py
Description: Utility functions for latents.
"""

from typing import List, Tuple, Union
import numpy as np
import torch


# ----------------------------------------------------------------------------
# Latent Normalization + Denormalization
# ----------------------------------------------------------------------------
def compute_norm_scale_and_bias(
        norm_std: float,
        norm_mean: float,
        data_dist_std: List[float],
        data_dist_mean: List[float],
        dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the scale and bias, so that we can normalize the data to the given mean and std
    (assuming the data is from a normal distribution).

    Args:
        norm_std (float): The standard deviation of the normalization.
        norm_mean (float): The mean of the normalization.
        data_dist_std (List[float]): The standard deviation of the data distribution (per channel).
        data_dist_mean (List[float]): The mean of the data distribution (per channel).
    """
    scale_np = np.float32(norm_std) / np.float32(data_dist_std)
    bias_np = (np.float32(norm_mean) - np.float32(data_dist_mean)) * scale_np
    return torch.tensor(scale_np, dtype=dtype), torch.tensor(bias_np, dtype=dtype)


def denormalize_latents(latents: Union[torch.Tensor, List[torch.Tensor]], scale: torch.Tensor,
                        bias: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    De-normalize latents, so they have the mean and standard deviation of the
    Auto-Encoder's posterior. This assumes that the Auto-Encoder's posterior is a Multivariate
    Diagonal Gaussian Distribution.

    `latents` - List[torch.Tensor] (c t h w) or torch.Tensor (b c t h w)
    `scale` - (c), torch.Tensor
    `bias` - (c), torch.Tensor
    """
    if isinstance(latents, torch.Tensor):
        if latents.ndim == 5:
            # Ensure scale and bias are on the same device and dtype as latents
            scale = scale.to(device=latents.device, dtype=latents.dtype)
            bias = bias.to(device=latents.device, dtype=latents.dtype)
            latents = latents - bias.reshape(1, -1, 1, 1, 1)
            latents = latents / scale.reshape(1, -1, 1, 1, 1)
            return latents  # (b c t h w)
        else:
            raise ValueError(f"Invalid Dimension of Latents: {latents.shape}")

    elif isinstance(latents, list):
        if not latents:
            raise ValueError("Input list 'latents' cannot be empty.")
        if latents[0].ndim == 4:
            # Ensure scale and bias are on the same device and dtype as the latents in the list
            scale = scale.to(device=latents[0].device, dtype=latents[0].dtype)
            bias = bias.to(device=latents[0].device, dtype=latents[0].dtype)
            for i in range(len(latents)):
                latents[i] = latents[i] - bias.reshape(-1, 1, 1, 1)
                latents[i] = latents[i] / scale.reshape(-1, 1, 1, 1)
            return latents  # List [torch.Tensor] (c t h w)
        else:
            raise ValueError(f"Invalid Dimension of Latents: {latents[0].shape}")

    else:
        raise ValueError(
            "Invalid input type for 'latents'. Expected torch.Tensor or List[torch.Tensor].")
