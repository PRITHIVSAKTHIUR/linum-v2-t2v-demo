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
File: text2video.py
Description: Text-to-Video DiT Model for inference.
"""

import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.amp as amp
from tqdm import tqdm
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file as load_safetensors

from linum_v2.utils.latents import (
    compute_norm_scale_and_bias,
    denormalize_latents,
)
from linum_v2.models.t5 import LinumT5EncoderModel
from linum_v2.models.dit import DiTModel
from linum_v2.models.vae import VideoVAE
from linum_v2.utils.guidance import MomentumBuffer, adaptive_projected_guidance
from linum_v2.utils.random import StackedRandomGenerator


# ----------------------------------------------------------------------------
# Text-to-Video Model
# ----------------------------------------------------------------------------
class Linum_v2_Text2Video(nn.Module):
    """
    Text-to-Video Generation Model for inference.
    """

    def __init__(
        self,
        text_len: int = 256,
        dim: int = 1664,
        ffn_dim: int = 8192,
        num_heads: int = 13,
        num_layers: int = 40,
        video_data_dist_mean: List[float] = [
            -0.0663, 0.0412, -0.5472, 0.2165, -0.3675, 0.2322, -0.2412, 0.0367,
            -0.2894, -0.4233, 0.2349, 0.5966, 0.0849, -1.2945, 1.0459, 0.5444
        ],
        video_data_dist_std: List[float] = [
            1.0406, 0.6302, 1.1242, 1.3156, 0.6202, 0.7880, 0.9484, 0.9519,
            1.2490, 0.8381, 1.0187, 0.6503, 0.6616, 0.5637, 1.2180, 0.9530
        ],
        norm_mean: float = 0.0,
        norm_std: float = 0.5,
        zero_init_head_and_bias: bool = True,
    ):
        super().__init__()

        # Compute scales and biases to normalize data to the given mean and std
        video_data_scale, video_data_bias = compute_norm_scale_and_bias(
            norm_std=norm_std,
            norm_mean=norm_mean,
            data_dist_mean=video_data_dist_mean,
            data_dist_std=video_data_dist_std,
            dtype=torch.float32,
        )
        self.register_buffer('video_data_scale', video_data_scale.to(torch.float32))
        self.register_buffer('video_data_bias', video_data_bias.to(torch.float32))

        # Store original config values for logging
        self.video_data_dist_mean = video_data_dist_mean
        self.video_data_dist_std = video_data_dist_std
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # Initialize the DiTModel
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.zero_init_head_and_bias = zero_init_head_and_bias
        self.patch_size = (1, 2, 2) # T x H x W
        self.model = DiTModel(
            patch_size=self.patch_size,
            text_len=text_len,
            in_dim=16, # 16 channel VAE
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            freq_dim=256,
            text_dim=4096, # T5 embedding dim
            out_dim=16, # 16 channel VAE
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            qk_norm=True,
            cross_attn_norm=True,
            epsilon=1e-6,
            zero_init_head_and_bias=zero_init_head_and_bias,
        )

    def __repr__(self):
        header = f"{'Attribute':<30} | {'Value':<60}"
        separator = '-' * 95

        repr_str = f"{header}\n{separator}\n"
        repr_str += f"{'norm_mean':<30} | {self.norm_mean:<60}\n"
        repr_str += f"{'norm_std':<30} | {self.norm_std:<60}\n"
        repr_str += f"{'video_data_dist_mean':<30} | {self.video_data_dist_mean!s:<60}\n"
        repr_str += f"{'video_data_dist_std':<30} | {self.video_data_dist_std!s:<60}\n"

        # DiT Parameters
        repr_str += "\n"
        repr_str += f"{'DiT Parameters':<30} | {'':<60}\n"
        repr_str += f"{'--------------------------------':<30} | {'':<60}\n"
        repr_str += f"{'dim':<30} | {self.dim:<60}\n"
        repr_str += f"{'ffn_dim':<30} | {self.ffn_dim:<60}\n"
        repr_str += f"{'num_heads':<30} | {self.num_heads:<60}\n"
        repr_str += f"{'num_layers':<30} | {self.num_layers:<60}\n"
        repr_str += f"{'zero_init_head_and_bias':<30} | {str(self.zero_init_head_and_bias):<60}\n"

        # Add total parameter count
        total_params = sum(p.numel() for p in self.parameters())
        params_billions = total_params / 1e9
        repr_str += f"{'total_parameters':<30} | {params_billions:.1f}B{' ':<60}\n"

        return repr_str

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        **kwargs: Any
    ) -> 'Linum_v2_Text2Video':
        """
        Loads pre-trained weights from a checkpoint file into a new model instance.

        Args:
            checkpoint_path (str):
                Path to the checkpoint file (.pt or .safetensors).
            **kwargs (Any):
                Additional keyword arguments to pass to the constructor.

        Returns:
            Linum_v2_Text2Video:
                An instance with weights loaded from the checkpoint.
        """
        # Support both safetensors and legacy .pt formats
        if checkpoint_path.endswith('.safetensors'):
            model_state_dict = load_safetensors(checkpoint_path, device='cpu')
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'state_dict' not in checkpoint:
                raise ValueError(
                    "Checkpoint does not have the expected structure: missing 'state_dict' key")
            model_state_dict = checkpoint['state_dict']

        instance = cls(**kwargs)
        # strict=False to ignore legacy image_data_scale/image_data_bias buffers
        instance.load_state_dict(model_state_dict, strict=False)

        return instance

    @torch.inference_mode()
    def generate(
        self,
        input_prompt: str,
        size: Tuple[int, int],
        frame_num: int,
        sampling_steps: int,
        guide_scale: float,
        n_prompt: str,
        t5_tokenizer_path: Optional[str] = None,
        t5_model_path: Optional[str] = None,
        vae_weights_path: Optional[str] = None,
        seeds: Optional[List[int]] = None,
        apg_momentum: float = -0.75,
        apg_eta: float = 0.0,
        apg_rescale: float = 20.0,
        device: str = 'cuda',
        quiet: bool = False,
        text_encoder: Optional[LinumT5EncoderModel] = None,
        vae: Optional[VideoVAE] = None,
    ) -> List[torch.Tensor]:
        """
        Generates video from text prompt for multiple seeds in parallel.

        Args:
            input_prompt (str):
                Text prompt for content generation.
            size (Tuple[int, int]):
                Controls video resolution, (height, width).
            frame_num (int):
                How many frames to sample from a video. The number should be 4n+1.
            sampling_steps (int):
                Number of sampling steps.
            guide_scale (float):
                Classifier-free guidance scale.
            n_prompt (str):
                Negative prompt for content exclusion.
            t5_tokenizer_path (Optional[str]):
                Path to T5 tokenizer. Required if text_encoder is not provided.
            t5_model_path (Optional[str]):
                Path to T5 model. Required if text_encoder is not provided.
            vae_weights_path (Optional[str]):
                Path to VAE weights. Required if vae is not provided.
            seeds (Optional[List[int]]):
                List of random seeds for noise generation.
            apg_momentum (float):
                Momentum coefficient for APG. Default -0.75.
            apg_eta (float):
                Scale factor for parallel component in APG. Default 0.0.
            apg_rescale (float):
                Rescaling factor for APG. Default 20.0
            device (str):
                Device to run generation on. Default 'cuda'.
            quiet (bool):
                Whether to suppress progress output. Default False.
            text_encoder (Optional[LinumT5EncoderModel]):
                Pre-loaded T5 text encoder. If provided, t5_tokenizer_path and
                t5_model_path are ignored. The caller is responsible for cleanup.
            vae (Optional[VideoVAE]):
                Pre-loaded VAE model. If provided, vae_weights_path is ignored.
                The caller is responsible for cleanup.

        Returns:
            List[torch.Tensor]:
                List of generated video frames tensors, one for each seed.
                Each tensor has dimensions: (C, T, H, W).
        """

        # Determine Target Shape
        VAE_N_CHANNELS, VAE_T_STRIDE, VAE_H_STRIDE, VAE_W_STRIDE = 16, 4, 8, 8
        target_shape: Tuple[int, int, int, int] = (
            VAE_N_CHANNELS,
            (frame_num - 1) // VAE_T_STRIDE + 1,
            size[0] // VAE_H_STRIDE,
            size[1] // VAE_W_STRIDE,
        )

        # Track ownership for cleanup (only clean up models we load ourselves)
        owns_text_encoder = text_encoder is None
        owns_vae = vae is None

        # Load T5 Encoder if not provided
        if text_encoder is None:
            if t5_model_path is None or t5_tokenizer_path is None:
                raise ValueError(
                    "Either text_encoder must be provided, or both "
                    "t5_model_path and t5_tokenizer_path must be specified."
                )
            text_encoder = LinumT5EncoderModel(
                checkpoint_path=t5_model_path,
                tokenizer_path=t5_tokenizer_path,
                dtype=torch.bfloat16,
                device=device,
                max_length=self.model.text_len,
            )

        # Load VAE if not provided
        if vae is None:
            if vae_weights_path is None:
                raise ValueError(
                    "Either vae must be provided, or vae_weights_path must be specified."
                )
            vae = VideoVAE(
                vae_pth=vae_weights_path,
                device=device,
                dtype=torch.bfloat16,
                normalize_latents=False,
            )

        # Generate Text Contexts
        batch_size = len(seeds)

        text_input: torch.Tensor = (text_encoder([input_prompt], device)[0]).repeat(batch_size, 1, 1)
        text_input_null: torch.Tensor = (text_encoder([n_prompt], device)[0]).repeat(batch_size, 1, 1)

        # Set frame_lens for generation (video-only, always multi-frame)
        frame_lens = torch.full(
            (batch_size,),
            target_shape[1],
            dtype=torch.int32,
            device=device,
        )

        arg_cond: Dict[str, Any] = {'text_input': text_input, 'frame_lens': frame_lens}
        arg_null: Dict[str, Any] = {'text_input': text_input_null, 'frame_lens': frame_lens}

        # Our Flow Matching Notation Convention:
        # - x_1 = Random sample from multivariate gaussian distribution (noise)
        # - x_0 = Latents of a video (data)
        # ------------------------------------------------------------------------------------------
        # Note, this is inverted from typical flow matching literature where x_0 = random sample,
        # x_1 = data, and we're interpolating from random noise to data.
        # We use this inverted convention to align with diffusers library scheduler implementations,
        # which rely on diffusion model nomenclature, wherein we start with data at x_0 and 
        # noise it over t steps to get to x_1.
        # ------------------------------------------------------------------------------------------
        # Generate x_t (t=1.0) (sample from multivariate gaussian distribution)
        stacked_generator = StackedRandomGenerator(device, seeds)
        x_t = stacked_generator.randn(
            (batch_size, target_shape[0], target_shape[1], target_shape[2], target_shape[3]),
            dtype=torch.bfloat16,
            device=device,
        )

        # Generate Samples in Parallel
        with amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Set Sampling Schedule and get timesteps (always shift=3.0, velocity prediction)
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=device)
            timesteps: torch.Tensor = sample_scheduler.timesteps

            # Initialize APG momentum buffer
            momentum_buffer = MomentumBuffer(apg_momentum)

            # Solve ODE
            progress_bar = tqdm(timesteps, disable=quiet)
            for i, t in enumerate(progress_bar):
                t = t.repeat(batch_size).to(device)

                # Get conditional and unconditional predictions
                v_cond: torch.Tensor = self.model(x_t, t=t, **arg_cond)
                v_null: torch.Tensor = self.model(x_t, t=t, **arg_null)

                # Apply APG (velocity prediction, spatial rescale only)
                pred_guided = adaptive_projected_guidance(
                    pred_cond=v_cond,
                    pred_other=v_null,
                    guidance_scale=guide_scale,
                    momentum_buffer=momentum_buffer,
                    eta=apg_eta,
                    rescale=apg_rescale,
                    spatial_rescale_only=True,
                )

                # Take sampling step
                x_t = sample_scheduler.step(
                    pred_guided,
                    timesteps[i],
                    x_t,
                    return_dict=False,
                )[0]

            # Denormalize Latents
            x_t = denormalize_latents(
                latents=x_t,
                scale=self.video_data_scale,
                bias=self.video_data_bias,
            )

            # Decode Latents + Normalize to 0-255 (RGB)
            samples: torch.Tensor = vae.decode(x_t)
            samples = ((samples + 1) * 127.5).clamp(0, 255)

            # Clean up models from GPU memory (only if we loaded them)
            if owns_text_encoder:
                text_encoder.model.cpu()
                del text_encoder
            if owns_vae:
                vae.model.cpu()
                del vae

            del text_input, text_input_null
            del x_t, sample_scheduler, timesteps
            del stacked_generator
            gc.collect()
            torch.cuda.empty_cache()

            return [s for s in samples]
