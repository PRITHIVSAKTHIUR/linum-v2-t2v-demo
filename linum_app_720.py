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

import os
import sys
import gc
import time
import random
import tempfile
import uuid
import torch
import torchvision.io
import gradio as gr
import spaces
import numpy as np
from pathlib import Path
from typing import Iterable, Optional
from huggingface_hub import hf_hub_download, snapshot_download

# Gradio Theme Imports
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# -----------------------------------------------------------------------------
# Import Linum Model
# -----------------------------------------------------------------------------
try:
    from linum_v2.models.text2video import Linum_v2_Text2Video
except ImportError:
    # Fallback to local path if package not installed globally
    sys.path.append(os.path.abspath("."))
    try:
        from linum_v2.models.text2video import Linum_v2_Text2Video
    except ImportError:
        print("CRITICAL: Could not import Linum_v2_Text2Video. Ensure the linum_v2 package is installed.")

# -----------------------------------------------------------------------------
# 1. OrangeRed Theme Configuration
# -----------------------------------------------------------------------------

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# ----------------------------------------------------------------------------
# 2. UI Components (RadioAnimated)
# ----------------------------------------------------------------------------

class RadioAnimated(gr.HTML):
    def __init__(self, choices, value=None, **kwargs):
        if not choices or len(choices) < 2:
            raise ValueError("RadioAnimated requires at least 2 choices.")
        if value is None:
            value = choices[0]

        uid = uuid.uuid4().hex[:8]
        group_name = f"ra-{uid}"

        inputs_html = "\n".join(
            f"""
            <input class="ra-input" type="radio" name="{group_name}" id="{group_name}-{i}" value="{c}">
            <label class="ra-label" for="{group_name}-{i}">{c}</label>
            """
            for i, c in enumerate(choices)
        )

        html_template = f"""
        <div class="ra-wrap" data-ra="{uid}">
          <div class="ra-inner">
            <div class="ra-highlight"></div>
            {inputs_html}
          </div>
        </div>
        """

        js_on_load = r"""
        (() => {
          const wrap = element.querySelector('.ra-wrap');
          const inner = element.querySelector('.ra-inner');
          const highlight = element.querySelector('.ra-highlight');
          const inputs = Array.from(element.querySelectorAll('.ra-input'));

          if (!inputs.length) return;

          const choices = inputs.map(i => i.value);

          function setHighlightByIndex(idx) {
            const n = choices.length;
            const pct = 100 / n;
            highlight.style.width = `calc(${pct}% - 6px)`;
            highlight.style.transform = `translateX(${idx * 100}%)`;
          }

          function setCheckedByValue(val, shouldTrigger=false) {
            const idx = Math.max(0, choices.indexOf(val));
            inputs.forEach((inp, i) => { inp.checked = (i === idx); });
            setHighlightByIndex(idx);

            props.value = choices[idx];
            if (shouldTrigger) trigger('change', props.value);
          }

          setCheckedByValue(props.value ?? choices[0], false);

          inputs.forEach((inp) => {
            inp.addEventListener('change', () => {
              setCheckedByValue(inp.value, true);
            });
          });
        })();
        """

        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            **kwargs
        )

# ----------------------------------------------------------------------------
# 3. Constants & Configuration
# ----------------------------------------------------------------------------

HF_REPO_360P = "Linum-AI/linum-v2-360p"
HF_REPO_720P = "Linum-AI/linum-v2-720p"

DIT_FILE = {
    "360p": "dit/360p.safetensors",
    "720p": "dit/720p.safetensors",
}
VAE_FILE = "vae/vae.safetensors"
T5_ENCODER_DIR = "t5/text_encoder"
T5_TOKENIZER_DIR = "t5/tokenizer"

FPS = 24
MAX_SEED = np.iinfo(np.int32).max

RESOLUTIONS = {
    "360p": (360, 640),
    "720p": (720, 1280),
}

# ----------------------------------------------------------------------------
# 4. Startup: Download ALL Weights Immediately
# ----------------------------------------------------------------------------

def preload_models():
    """Downloads all necessary weights for both resolutions at app startup."""
    print("=" * 60)
    print("STARTUP: Pre-downloading all model files...")
    print("=" * 60)
    
    paths = {
        "360p": {},
        "720p": {}
    }

    # 1. Download 720p Assets
    print(f"Downloading assets from {HF_REPO_720P}...")
    paths["720p"]["dit"] = hf_hub_download(repo_id=HF_REPO_720P, filename=DIT_FILE["720p"])
    paths["720p"]["vae"] = hf_hub_download(repo_id=HF_REPO_720P, filename=VAE_FILE)
    
    t5_enc_720 = snapshot_download(repo_id=HF_REPO_720P, allow_patterns=f"{T5_ENCODER_DIR}/*")
    paths["720p"]["t5_encoder"] = os.path.join(t5_enc_720, T5_ENCODER_DIR)
    
    t5_tok_720 = snapshot_download(repo_id=HF_REPO_720P, allow_patterns=f"{T5_TOKENIZER_DIR}/*")
    paths["720p"]["t5_tokenizer"] = os.path.join(t5_tok_720, T5_TOKENIZER_DIR)

    # 2. Download 360p Assets
    print(f"Downloading DiT from {HF_REPO_360P}...")
    paths["360p"]["dit"] = hf_hub_download(repo_id=HF_REPO_360P, filename=DIT_FILE["360p"])
    
    # Reuse VAE/T5 from 720p download
    paths["360p"]["vae"] = paths["720p"]["vae"] 
    paths["360p"]["t5_encoder"] = paths["720p"]["t5_encoder"]
    paths["360p"]["t5_tokenizer"] = paths["720p"]["t5_tokenizer"]

    print("=" * 60)
    print("STARTUP: All models downloaded and ready.")
    print("=" * 60)
    return paths

# Execute download immediately
CACHED_PATHS = preload_models()

# Global variable to hold the loaded model in VRAM
class ModelContainer:
    def __init__(self):
        self.model = None
        self.loaded_resolution = None

MODEL_CONTAINER = ModelContainer()

# ----------------------------------------------------------------------------
# 5. Inference Logic
# ----------------------------------------------------------------------------

def calc_timeout(prompt, negative_prompt, resolution, duration, seed, randomize_seed, cfg, num_steps, apg_rescale, gpu_duration, progress=None):
    """
    Returns the GPU duration specified by the user.
    """
    try:
        return int(gpu_duration)
    except:
        return 120

@spaces.GPU(duration=calc_timeout)
def generate_video(
    prompt: str,
    negative_prompt: str,
    resolution: str,
    duration: float,
    seed: int,
    randomize_seed: bool,
    cfg: float,
    num_steps: int,
    apg_rescale: float,
    gpu_duration: int,
    progress=gr.Progress(track_tqdm=True)
):
    global MODEL_CONTAINER, CACHED_PATHS

    # 1. Seed Handling
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    print(f"Generating: {resolution}, {duration}s, Seed: {seed}, Timeout: {gpu_duration}s")

    # 2. Model Loading / Swapping
    if MODEL_CONTAINER.loaded_resolution != resolution or MODEL_CONTAINER.model is None:
        print(f"Swapping model to {resolution}...")
        
        if MODEL_CONTAINER.model is not None:
            del MODEL_CONTAINER.model
            gc.collect()
            torch.cuda.empty_cache()
            MODEL_CONTAINER.model = None

        checkpoint = CACHED_PATHS[resolution]["dit"]
        model = Linum_v2_Text2Video.from_pretrained(checkpoint_path=checkpoint)
        model = model.to('cuda').eval()
        
        MODEL_CONTAINER.model = model
        MODEL_CONTAINER.loaded_resolution = resolution
        print("Model loaded.")

    # 3. Setup Frames
    height, width = RESOLUTIONS[resolution]
    num_frames = int(FPS * duration) + 1
    if (num_frames - 1) % FPS != 0:
        num_frames = ((num_frames - 1) // FPS) * FPS + 1

    # 4. Generation
    try:
        current_paths = CACHED_PATHS[resolution]
        
        with torch.inference_mode():
            video_tensors = MODEL_CONTAINER.model.generate(
                input_prompt=prompt,
                size=(height, width),
                frame_num=num_frames,
                sampling_steps=num_steps,
                guide_scale=cfg,
                n_prompt=negative_prompt,
                t5_tokenizer_path=current_paths["t5_tokenizer"],
                t5_model_path=current_paths["t5_encoder"],
                vae_weights_path=current_paths["vae"],
                seeds=[seed],
                apg_rescale=apg_rescale,
                device='cuda',
                quiet=False, 
            )

        # 5. Saving
        video_tensor = video_tensors[0] # (C, T, H, W)
        video_tensor = video_tensor.permute(1, 2, 3, 0) # (T, H, W, C)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            output_path = tmpfile.name

        torchvision.io.write_video(
            str(output_path),
            video_tensor.cpu(),
            fps=FPS,
            video_codec="h264",
            options={"crf": "18"},
        )
        
        return output_path, seed

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {str(e)}")


# -----------------------------------------------------------------------------
# 6. UI Helpers
# -----------------------------------------------------------------------------

def apply_duration(val: str):
    """Converts '3s' -> 3.0"""
    return float(val.replace("s", ""))

def apply_gpu_duration(val: str):
    """Converts '120' -> 120"""
    return int(val)

# -----------------------------------------------------------------------------
# 7. Gradio Application
# -----------------------------------------------------------------------------

css = """
    #col-container {
        margin: 0 auto;
        max-width: 1200px;
    }
    #step-column {
        padding: 20px;
        border-radius: 12px;
        background: var(--background-fill-secondary);
        border: 1px solid var(--border-color-primary);
        margin-bottom: 20px;
    }
    .button-gradient {
        background: linear-gradient(90deg, #FF4500, #E63E00);
        border: none;
        color: white;
        font-weight: bold;
    }
    
    /* RadioAnimated CSS */
    .ra-wrap{ width: fit-content; }
    .ra-inner{
      position: relative; display: inline-flex; align-items: center; gap: 0; padding: 6px;
      background: var(--neutral-200); border-radius: 9999px; overflow: hidden;
    }
    .ra-input{ display: none; }
    .ra-label{
      position: relative; z-index: 2; padding: 8px 16px;
      font-family: inherit; font-size: 14px; font-weight: 600;
      color: var(--neutral-500); cursor: pointer; transition: color 0.2s; white-space: nowrap;
    }
    .ra-highlight{
      position: absolute; z-index: 1; top: 6px; left: 6px;
      height: calc(100% - 12px); border-radius: 9999px;
      background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      transition: transform 0.2s, width 0.2s;
    }
    .ra-input:checked + .ra-label{ color: black; }
    
    /* Dark mode adjustments for Radio */
    .dark .ra-inner { background: var(--neutral-800); }
    .dark .ra-label { color: var(--neutral-400); }
    .dark .ra-highlight { background: var(--neutral-600); }
    .dark .ra-input:checked + .ra-label { color: white; }

    #main-title h1 { font-size: 2.2em !important; }
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        
        # Header
        gr.Markdown("# **Linum-v2-720p-HF-Demo**", elem_id="main-title")
        gr.Markdown("Generate high-quality 2D animation style videos using the Linum-v2 model.")
        
        with gr.Row():
            # Left Column: Inputs
            with gr.Column(elem_id="step-column", scale=4):
                prompt = gr.Textbox(
                    label="Prompt",
                    value="In a charming hand-drawn 2D animation style, a rust-orange fox with cream chest fur grips a cherry-red steering wheel. Stylized trees and pastel houses whoosh past in smooth parallax.",
                    lines=4,
                    placeholder="Describe your animation..."
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="",
                    lines=1,
                    placeholder="Low quality, blurry, distorted..."
                )
                
                with gr.Accordion("Advanced Settings", open=True):
                    with gr.Row():
                        resolution = gr.Dropdown(
                            label="Resolution", 
                            choices=["360p", "720p"], 
                            value="720p",
                            info="720p requires more VRAM."
                        )
                        
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Duration**")
                            radioanimated_duration = RadioAnimated(
                                choices=["3s", "5s", "10s"],
                                value="3s",
                                elem_id="radioanimated_duration"
                            )
                            # Hidden component to store the actual float value
                            duration = gr.Number(value=3.0, visible=False)
                            
                        with gr.Column():
                            gr.Markdown("**GPU Duration**")
                            radioanimated_gpu_duration = RadioAnimated(
                                choices=["90", "120", "180", "240", "300"],
                                value="120",
                                elem_id="radioanimated_gpu_duration"
                            )
                            # Hidden component to store the actual int value
                            gpu_duration_state = gr.Number(value=120, visible=False)
                    
                    with gr.Row():
                        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, value=20, step=1)
                        randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    
                    with gr.Row():
                        cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=20.0, value=10.0, step=0.5)
                        num_steps = gr.Slider(label="Steps", minimum=10, maximum=100, value=50, step=1)
                        apg_rescale = gr.Slider(label="APG Rescale", minimum=0.0, maximum=50.0, value=20.0, step=1.0)

                generate_btn = gr.Button("Generate Video", variant="primary", elem_classes="button-gradient")

            # Right Column: Output
            with gr.Column(elem_id="step-column", scale=5):
                output_video = gr.Video(label="Generated Output", autoplay=True, height=512)
                output_seed_display = gr.Number(label="Used Seed", interactive=False)

    # Event Wiring for Custom Radios
    radioanimated_duration.change(
        fn=apply_duration, 
        inputs=[radioanimated_duration], 
        outputs=[duration], 
        api_visibility="private"
    )
    
    radioanimated_gpu_duration.change(
        fn=apply_gpu_duration, 
        inputs=[radioanimated_gpu_duration], 
        outputs=[gpu_duration_state], 
        api_visibility="private"
    )

    # Main Generation Event
    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt, negative_prompt, resolution, duration, seed, 
            randomize_seed, cfg, num_steps, apg_rescale, gpu_duration_state
        ],
        outputs=[output_video, output_seed_display]
    )

    # Examples
    gr.Examples(
        examples=[
            [
                "In a charming hand-drawn 2D animation style, a rust-orange fox with cream chest fur grips a cherry-red steering wheel. Stylized trees and pastel houses whoosh past in smooth parallax.",
                "", "720p", 3.0, 20, False, 10.0, 50, 20.0, 120
            ],
            [
                "Cyberpunk city street at night, neon rain, anime style, highly detailed, 4k.",
                "blurry, low quality", "360p", 3.0, 42, True, 10.0, 50, 20.0, 120
            ]
        ],
        fn=generate_video,
        inputs=[
            prompt, negative_prompt, resolution, duration, seed, 
            randomize_seed, cfg, num_steps, apg_rescale, gpu_duration_state
        ],
        outputs=[output_video, output_seed_display],
        cache_examples=False,
        label="Examples"
    )

if __name__ == "__main__":
    demo.queue().launch(theme=orange_red_theme, css=css)