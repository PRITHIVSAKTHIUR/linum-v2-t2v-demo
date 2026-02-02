# **linum-v2-t2v-demo**

> A Gradio-based demonstration for the Linum-AI/linum-v2 model (360p and 720p variants) for text-to-video generation. Generates high-quality 2D animation style videos from prompts, with support for resolutions up to 720p, customizable durations (3s-10s), and parameters like CFG scale, steps, and APG rescale. Features pre-downloading of all weights at startup for faster inference, lazy model swapping, custom OrangeRed theme, and animated radio selectors for intuitive UI.

## Features
- **Resolutions**: 360p (360x640) and 720p (720x1280) for varying quality and VRAM needs.
- **Durations**: Selectable options (3s, 5s, 10s) with FPS fixed at 24.
- **GPU Duration Settings**: Configurable timeouts (90s-300s) for Hugging Face Spaces compatibility.
- **Pre-Downloading**: All model weights (DiT, VAE, T5 encoder/tokenizer) downloaded at startup to avoid delays.
- **Lazy Model Loading**: Switches models between resolutions with VRAM clearing for efficiency.
- **Inference Parameters**: Adjustable CFG scale (1-20), steps (10-100), APG rescale (0-50), seed randomization.
- **Custom UI**: OrangeRedTheme with animated radio buttons for duration/GPU settings, responsive layout.
- **Video Output**: Generates MP4 videos with h264 codec, autoplay support in UI.
- **Examples**: Curated prompts for quick testing.
- **Queueing**: Up to default Gradio queue limits for concurrent jobs.

## Prerequisites
- Python 3.10 or higher.
- CUDA-compatible GPU (required for efficient inference; 720p needs substantial VRAM).
- Stable internet for initial model downloads from Hugging Face.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/linum-v2-t2v-demo.git
   cd linum-v2-t2v-demo
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   **requirements.txt content:**
   ```
   # Core ML (PyTorch 2.7.1 with CUDA 12.8)
   torch==2.7.1
   torchvision>=0.22.0
   triton # torch 2.7.1 bundled triton is incompatible

   # Hugging Face ecosystem
   diffusers==0.34.0
   transformers==4.53.2
   huggingface-hub>=0.20.0
   safetensors>=0.4.0
   sentencepiece==0.2.0
   protobuf>=3.20.0

   # Tensor operations
   einops==0.8.1
   numpy==2.1.1

   # Video I/O
   av>=14.4.0

   # Utilities
   tqdm
   packaging>=25.0

   # Flash Attention 3 build dependency (optional, Hopper GPUs only)
   ninja

   # SDK
   gradio==6.3.0
   ```
3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860`.

## Usage
1. **Enter Prompt**: Describe the video scene (e.g., animation style, actions).
2. **Negative Prompt (Optional)**: Specify elements to avoid (e.g., "blurry").
3. **Select Resolution**: Choose 360p or 720p.
4. **Choose Duration**: Use animated radio for 3s, 5s, or 10s.
5. **GPU Duration**: Select timeout for generation (affects Spaces runtime).
6. **Advanced Settings**: Adjust seed, CFG, steps, APG rescale.
7. **Generate**: Click "Generate Video" to produce output.

### Supported Resolutions
| Resolution | Dimensions | Use Case |
|------------|------------|----------|
| 360p      | 360x640   | Faster, lower VRAM |
| 720p      | 720x1280  | Higher quality, more detail |

## Examples
| Prompt | Negative Prompt | Resolution | Duration | Seed | CFG | Steps | APG Rescale |
|--------|-----------------|------------|----------|------|-----|-------|-------------|
| "In a charming hand-drawn 2D animation style, a rust-orange fox with cream chest fur grips a cherry-red steering wheel. Stylized trees and pastel houses whoosh past in smooth parallax." | "" | 720p | 3s | 20 | 10.0 | 50 | 20.0 |
| "Cyberpunk city street at night, neon rain, anime style, highly detailed, 4k." | "blurry, low quality" | 360p | 3s | 42 | 10.0 | 50 | 20.0 |

## Troubleshooting
- **Model Loading**: All weights download at startup; monitor console for progress.
- **OOM Errors**: Use 360p for lower VRAM; clear cache manually if needed.
- **Generation Fails**: Ensure prompt is descriptive; check for CUDA availability.
- **Timeout Issues**: Increase GPU duration for longer videos or complex prompts.
- **No Output**: Verify inputs; review console for errors like import failures.

## Contributing
Contributions welcome! Add new features like more resolutions, enhance UI, or optimize inference. Submit pull requests via the repository.

Repository: [https://github.com/PRITHIVSAKTHIUR/linum-v2-t2v-demo.git](https://github.com/PRITHIVSAKTHIUR/linum-v2-t2v-demo.git)

## License
Apache License 2.0. See [LICENSE](LICENSE) for details.
Built by Prithiv Sakthi. Report issues via the repository.
