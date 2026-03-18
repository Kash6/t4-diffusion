# t4-diffusion

TensorRT-optimized Stable Diffusion for NVIDIA T4 GPUs on Google Colab's free tier.

Provides INT8 quantization, TensorRT compilation, feature caching, and VRAM monitoring. Designed to achieve 1.5-2x speedup while staying within the 15.6GB VRAM constraint of T4 GPUs.

## Features

- **INT8 Quantization** via nvidia-modelopt with SmoothQuant algorithm
- **TensorRT Compilation** for optimized inference
- **Feature Caching** for additional acceleration
- **VRAM Monitoring** with 15.6GB T4 limit enforcement
- **Property-Based Testing** for correctness guarantees
- **Easy-to-use API** compatible with HuggingFace diffusers

## Supported Models

- **SDXL-Turbo** (`stabilityai/sdxl-turbo`) - 4 steps, best for real-time
- **Stable Diffusion 1.5** (`runwayml/stable-diffusion-v1-5`) - 20 steps, good balance

## Target Hardware

- NVIDIA T4 GPU (sm_75, 15.6GB VRAM)
- Google Colab Free Tier

## Installation

### Google Colab (CUDA 13.x)

```bash
# Install CUDA 13 compatible TensorRT packages
pip install tensorrt-cu13 tensorrt-lean-cu13

# Install torch-tensorrt for TensorRT compilation
pip install torch-tensorrt

# Install nvidia-modelopt 0.39+ for CUDA 13
pip install nvidia-modelopt>=0.39.0

# Install t4-diffusion
pip install git+https://github.com/Kash6/t4-diffusion.git
```

### Local Development (CUDA 12.x)

```bash
git clone https://github.com/Kash6/t4-diffusion.git
cd t4-diffusion
pip install -e ".[dev,tensorrt-cuda12]"
```

## Quick Start

```python
from diffusion_trt import OptimizedPipeline, PipelineConfig

config = PipelineConfig(
    model_id="stabilityai/sdxl-turbo",
    enable_int8=True,           # INT8 quantization
    enable_caching=True,        # Feature caching
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

Expected benchmarks on NVIDIA T4 (Google Colab Free Tier):

| Configuration | FP16 Baseline | INT8 TensorRT | Speedup |
|--------------|---------------|---------------|---------|
| SDXL-Turbo @ 512×512, 4 steps | ~1.5s | ~0.8-1.0s | 1.5-2x |
| SD 1.5 @ 512×512, 20 steps | ~4.0s | ~2.5-3.0s | 1.3-1.6x |

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

### For CUDA 13.x (Google Colab March 2026+)
- tensorrt-cu13
- tensorrt-lean-cu13
- torch-tensorrt
- nvidia-modelopt >= 0.39.0

### For CUDA 12.x
- tensorrt
- torch-tensorrt >= 2.1.0
- nvidia-modelopt >= 0.15.0

## License

MIT License
