# t4-diffusion

TensorRT-optimized Stable Diffusion for NVIDIA T4 GPUs on Google Colab's free tier.

Combines INT8 Post-Training Quantization via TensorRT Model Optimizer with DeepCache-style feature caching to achieve up to ~2x speedups on selected pipelines while staying within the 15.6GB VRAM constraint.

## Features

- **INT8 Quantization**: Post-Training Quantization using TensorRT Model Optimizer with SmoothQuant
- **Feature Caching**: DeepCache-style caching for training-free acceleration
- **VRAM Optimization**: Designed for T4 GPU's 15.6GB VRAM limit with automatic monitoring
- **Fixed-Resolution Presets**: Pre-configured engine presets for optimal T4 performance
- **Property-Based Testing**: Correctness guarantees via Hypothesis-based property tests
- **Easy Integration**: API compatible with HuggingFace diffusers pipelines

## Supported Models

- **SDXL-Turbo** (`stabilityai/sdxl-turbo`) - 4 steps, best for real-time
- **Stable Diffusion 1.5** (`runwayml/stable-diffusion-v1-5`) - 20 steps, good balance

## Target Hardware

- NVIDIA T4 GPU (sm_75, 15.6GB VRAM, INT8 Tensor Cores)
- Google Colab Free Tier

## Installation

```bash
pip install t4-diffusion
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
    enable_int8=True,
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

### Using Presets

```python
from diffusion_trt import get_preset, OptimizedPipeline

# Get recommended preset for T4
config = get_preset("SDXL_TURBO_512")
pipeline = OptimizedPipeline.from_pretrained(config.model_id, config=config)

# Or auto-select based on VRAM
from diffusion_trt import get_recommended_preset
preset_name = get_recommended_preset(max_vram_gb=15.6, prefer_quality=False)
config = get_preset(preset_name)
```

## Performance

Benchmarks on NVIDIA T4 (Google Colab Free Tier):

| Configuration | Latency | Speedup vs FP16 | VRAM |
|--------------|---------|-----------------|------|
| SD 1.5 @ 512×512, 20 steps, INT8 | ~3.5s | ~1.5x | ~6 GB |
| SDXL-Turbo @ 512×512, 4 steps, INT8 | ~0.9s | ~1.5x | ~8 GB |

*Note: Actual speedups vary by model, resolution, and baseline. Target is ≥1.5x speedup.*

## Available Presets

| Preset | Model | Resolution | Steps | Est. VRAM |
|--------|-------|------------|-------|-----------|
| `SD15_512` | SD 1.5 | 512×512 | 20 | 6 GB |
| `SD15_768` | SD 1.5 | 768×768 | 20 | 8.5 GB |
| `SDXL_TURBO_512` | SDXL-Turbo | 512×512 | 4 | 8 GB |
| `SDXL_768` | SDXL-Turbo | 768×768 | 4 | 10.5 GB |
| `SDXL_1024` | SDXL-Turbo | 1024×1024 | 4 | 14 GB |

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
- TensorRT >= 8.6
- Torch-TensorRT >= 2.1.0

## License

MIT License
