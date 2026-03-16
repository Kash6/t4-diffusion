# TensorRT Diffusion Model Optimization Pipeline

A TensorRT-based optimization pipeline for diffusion models targeting NVIDIA T4 GPUs on Google Colab's free tier. Combines INT8 Post-Training Quantization via TensorRT Model Optimizer with DeepCache/X-Slim style feature caching to achieve 2-3x speedups while staying within the 15.6GB VRAM constraint.

## Features

- **INT8 Quantization**: Post-Training Quantization using TensorRT Model Optimizer with SmoothQuant
- **Feature Caching**: DeepCache/X-Slim style caching for training-free acceleration
- **VRAM Optimization**: Designed for T4 GPU's 15.6GB VRAM limit
- **Easy Integration**: API compatible with HuggingFace diffusers pipelines

## Target Models

- SDXL-Turbo (4 steps)
- Stable Diffusion 1.5 (20 steps)

## Target Hardware

- NVIDIA T4 GPU (sm_75, 15.6GB VRAM, INT8 Tensor Cores)
- Google Colab Free Tier

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from diffusion_trt import OptimizedPipeline, PipelineConfig

config = PipelineConfig(
    model_id="stabilityai/sdxl-turbo",
    enable_int8=True,
    enable_caching=True,
    cache_interval=3,
    num_inference_steps=4
)

# Load and optimize
pipeline = OptimizedPipeline.from_pretrained(config.model_id, config=config)

# Generate image
image = pipeline("A photo of a cat wearing sunglasses")[0]
image.save("output.png")
```

## Performance Targets

| Model | Steps | Baseline | Optimized | Speedup |
|-------|-------|----------|-----------|---------|
| SD 1.5 | 20 | ~8s | <4s | 2x+ |
| SDXL-Turbo | 4 | ~2s | <1s | 2x+ |

## Requirements

- Python >= 3.10
- CUDA-capable GPU with compute capability >= 7.5
- PyTorch >= 2.1.0
- TensorRT >= 8.6
- Torch-TensorRT >= 2.1.0

## License

MIT License
