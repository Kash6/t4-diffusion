# t4-diffusion

Stable Diffusion optimized for NVIDIA T4 GPUs on Google Colab's free tier.

Provides memory-optimized FP16 inference with VRAM monitoring, feature caching, and graceful fallbacks. Designed to work within the 15.6GB VRAM constraint of T4 GPUs.

## Current Status

**Working Features:**
- FP16 inference with memory optimizations (attention slicing, VAE tiling)
- VRAM monitoring with 15.6GB T4 limit enforcement
- Feature caching infrastructure
- Property-based testing for correctness guarantees
- Easy-to-use API compatible with HuggingFace diffusers

**Planned Features (blocked by Colab CUDA compatibility):**
- INT8 Post-Training Quantization via TensorRT Model Optimizer
- TensorRT compilation for additional speedup

> **Note:** As of March 2026, Google Colab uses CUDA 13.x which has compatibility issues with TensorRT/nvidia-modelopt packages (built for CUDA 12.x). The pipeline automatically falls back to optimized FP16 inference when these dependencies aren't available.

## Supported Models

- **SDXL-Turbo** (`stabilityai/sdxl-turbo`) - 4 steps, best for real-time
- **Stable Diffusion 1.5** (`runwayml/stable-diffusion-v1-5`) - 20 steps, good balance

## Target Hardware

- NVIDIA T4 GPU (sm_75, 15.6GB VRAM)
- Google Colab Free Tier

## Installation

```bash
pip install git+https://github.com/Kash6/t4-diffusion.git
```

For development:
```bash
git clone https://github.com/Kash6/t4-diffusion.git
cd t4-diffusion
pip install -e ".[dev]"
```

## Quick Start

```python
from diffusion_trt import OptimizedPipeline, PipelineConfig

config = PipelineConfig(
    model_id="stabilityai/sdxl-turbo",
    enable_int8=True,           # Falls back to FP16 if unavailable
    enable_caching=True,
    num_inference_steps=4,
    guidance_scale=0.0,
)

# Load and optimize
pipeline = OptimizedPipeline.from_pretrained(config.model_id, config=config)

# Generate image
image = pipeline("A photo of a cat wearing sunglasses")[0]
image.save("output.png")
```

## Performance

Current benchmarks on NVIDIA T4 (Google Colab Free Tier) with FP16:

| Configuration | Latency | Throughput | VRAM |
|--------------|---------|------------|------|
| SDXL-Turbo @ 512×512, 4 steps | ~1.5s | ~0.65 img/s | ~12 GB |

*With INT8/TensorRT (when available): expect ~1.5-2x speedup*

## Colab Notebook

Try it on Google Colab: [t4_diffusion_demo.ipynb](notebooks/t4_diffusion_demo.ipynb)

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features including:

- **v0.2.0**: LCM-LoRA support for 2-8 step generation
- **v0.3.0**: SDXL-Lightning support
- **v0.4.0**: Hyper-SD support

## Requirements

- Python >= 3.10
- CUDA-capable GPU with compute capability >= 7.5 (T4, RTX 20xx+)
- PyTorch >= 2.1.0

Optional (for INT8/TensorRT when CUDA compatible):
- TensorRT >= 8.6
- Torch-TensorRT >= 2.1.0
- nvidia-modelopt >= 0.15.0

## License

MIT License
