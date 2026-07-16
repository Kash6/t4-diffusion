# t4-diffusion

INT8-optimized Stable Diffusion for NVIDIA T4 GPUs on Google Colab's free tier.

Provides INT8 quantization, feature caching, and VRAM monitoring, with an
easy-to-use API compatible with HuggingFace `diffusers`. Two INT8 backends are
available depending on how much host RAM you have to work with — see
[Backends](#backends) below.

## Features

- **INT8 Quantization** — two interchangeable backends (Quanto and TensorRT) with different memory/speed tradeoffs, see below
- **Feature Caching** for additional acceleration
- **VRAM Monitoring** with 15.6GB T4 limit enforcement
- **Property-Based Testing** for correctness guarantees
- **Easy-to-use API** compatible with HuggingFace diffusers

## Backends

This project supports two INT8 backends. They trade off differently between
memory safety and speedup, and the right choice depends on how much **host
RAM** (not VRAM) you have available — this is the actual constraint on free
Colab, not GPU compute or GPU memory.

| | `quanto` (default) | `tensorrt` (opt-in) |
|---|---|---|
| Library | [optimum-quanto](https://github.com/huggingface/optimum-quanto) | nvidia-modelopt + TensorRT |
| How it works | Quantizes + freezes UNet weights in-place, pure PyTorch ops | ONNX export → native TensorRT INT8 engine build |
| Extra host RAM needed | ~0.1-1GB | **+4GB or more** during ONNX export alone |
| Works on free Colab (~12GB RAM)? | Yes | Not reliably — see below |
| Speedup vs FP16 (measured, RTX 5080, SD1.5, 20 steps) | 0.86x (1100ms vs 947ms baseline — slower, not faster) | 1.95x (486ms vs 947ms baseline) |
| Best for | Free-tier Colab, laptops, any RAM-constrained box | Colab Pro, local GPUs with 16GB+ system RAM |

**Why `quanto` is the default — memory safety, not speed:** the free-tier
Colab T4 has plenty of VRAM (15.6GB) but only about 12GB of *host* RAM. The
TensorRT path's ONNX export step alone can spike host RAM usage by 4GB or
more on top of a 7-8GB pipeline baseline — that reliably exceeds Colab's
ceiling and crashes the kernel with no traceback. Quanto quantizes the UNet
in-place with no export or engine-build step, so there's no
tracing/serialization phase to spike during, and it's verified working
end-to-end on an actual free-tier T4 (see Performance below).

Be aware, though: on the hardware we measured (RTX 5080), quanto's
weight-only INT8 quantization was measurably **slower** than plain FP16
(0.86x), not faster — likely because that GPU's kernels don't have an
optimized INT8 GEMM path for this quantization scheme, so it pays
dequantization overhead without a compute win. Quanto is the default
because it's the only backend that reliably *runs* on free-tier Colab, not
because it's faster there. If you need real speedup, use `tensorrt`.

**Why `tensorrt` is still here — real speedup, but needs more RAM:** it
measured 1.95x faster than FP16 (486ms vs 947ms) on the same local RTX 5080,
using genuine INT8 tensor-core kernels via a compiled engine. If you have
host RAM to spare — Colab Pro's higher-RAM runtimes, your own GPU, or a
cloud GPU box — it's the better choice. Set `backend="tensorrt"` explicitly
to opt in.

```python
from diffusion_trt import OptimizedPipeline, PipelineConfig

# Default: safe on free-tier Colab
config = PipelineConfig(model_id="stabilityai/sdxl-turbo", backend="quanto")

# Opt-in: better speedup, needs more host RAM (Colab Pro / local GPU)
config = PipelineConfig(model_id="stabilityai/sdxl-turbo", backend="tensorrt")
```

## Supported Models

- **SDXL-Turbo** (`stabilityai/sdxl-turbo`) - 4 steps, best for real-time
- **Stable Diffusion 1.5** (`runwayml/stable-diffusion-v1-5`) - 20 steps, good balance

## Target Hardware

- NVIDIA T4 GPU (sm_75, 15.6GB VRAM), Google Colab Free Tier — `quanto` backend
- Any CUDA GPU with more host RAM headroom (Colab Pro, local GPUs) — `tensorrt` backend also viable

## Installation

### Google Colab (free tier, default `quanto` backend)

```bash
pip install git+https://github.com/Kash6/t4-diffusion.git
```

`optimum-quanto` is a core dependency and installs automatically. No
TensorRT/ONNX packages are needed for the default backend.

### Opt-in TensorRT backend (Colab Pro / local GPU with more host RAM)

```bash
# Install t4-diffusion with the tensorrt extra
pip install "git+https://github.com/Kash6/t4-diffusion.git#egg=diffusion-trt[tensorrt]"

# Plus TensorRT + ONNX packages (versions drift across Colab sessions —
# install without pinning exact torch/torchvision versions):
pip install tensorrt tensorrt-lean onnx onnxscript
```

Do **not** pin `torch`/`torchvision` versions when installing on Colab — the
shipped versions change per session and pinning causes ABI mismatches (e.g.
`torchvision::nms does not exist`). Pinning `transformers<4.46` and
`diffusers<0.31` is fine and recommended for `nvidia-modelopt` compatibility.

### Local Development

```bash
git clone https://github.com/Kash6/t4-diffusion.git
cd t4-diffusion
pip install -e ".[dev]"          # quanto backend (default)
pip install -e ".[dev,tensorrt]" # + opt-in tensorrt backend
```

## Quick Start

```python
from diffusion_trt import OptimizedPipeline, PipelineConfig

config = PipelineConfig(
    model_id="stabilityai/sdxl-turbo",
    enable_int8=True,            # INT8 quantization
    backend="quanto",            # default; use "tensorrt" if you have the RAM for it
    enable_caching=True,         # Feature caching
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

**Measured on an actual free-tier Colab T4** (`quanto` backend, SDXL-Turbo,
4 steps, 512x512):

| Metric | Value |
|---|---|
| Peak VRAM (model load) | 6.62 GB |
| Peak VRAM (inference) | 5.16 GB |
| Latency (mean) | 2158 ms |
| Latency (p50 / p95) | 2146 ms / 2354 ms |
| Throughput | 0.46-0.51 img/s |
| Available VRAM headroom | 11.34 GB (of 15.6 GB) |

Measured locally (RTX 5080, SD1.5, 20 steps, 512x512), including the FP16
baseline for comparison:

| Backend | Latency | Throughput | Peak VRAM | Speedup vs FP16 |
|---|---|---|---|---|
| FP16 (no quantization) | 947ms | 1.06 img/s | 2.64GB | 1.0x (baseline) |
| `quanto` | 1100ms | 0.91 img/s | 1.91GB | 0.86x (slower) |
| `tensorrt` | 486ms | 2.06 img/s | 1.03GB (1.8GB engine) | 1.95x |

The 5080 numbers are faster in absolute terms than the T4 numbers above
because a 5080 has far more raw compute than a T4 — that part is expected.
What's less expected: `quanto` was *slower* than doing nothing (plain FP16)
on this GPU. It still reduces VRAM usage (1.91GB vs 2.64GB), and it's the
only backend that reliably runs on free-tier Colab, but don't expect it to
be a speed win by itself — see the Backends section above for why. The
`tensorrt` backend has not yet been re-verified on an actual T4 (it needs
more host RAM than free-tier Colab provides), but its host-RAM behavior is
well understood and its engine build succeeds reliably on any machine with
sufficient RAM headroom (e.g. Colab Pro, local/cloud GPUs).

## Notebooks

- **Free-tier Colab**: [t4.ipynb](notebooks/t4.ipynb) — uses the default `quanto` backend, verified working end-to-end on an actual free-tier T4.
- **Colab Pro / enterprise / your own GPU**: [t4_pro.ipynb](notebooks/t4_pro.ipynb) — for Colab Pro, your own T4 (or better), GPU marketplace clouds, AWS SageMaker Studio Lab, or any environment with more host RAM. Defaults to the `tensorrt` backend for a larger speedup.

## Requirements

- Python >= 3.10
- CUDA-capable GPU with compute capability >= 7.5 (T4, RTX 20xx+)
- PyTorch >= 2.1.0
- `optimum-quanto` >= 0.2.0 (installed automatically, default backend)

### Only if using the opt-in `tensorrt` backend
- tensorrt / tensorrt-lean
- onnx, onnxscript
- nvidia-modelopt >= 0.15.0 (CUDA 12.x) or >= 0.39.0 (CUDA 13.x)
- Significantly more host RAM than the ~12GB free-tier Colab provides is
  recommended — see [Backends](#backends) above.

## License

MIT License
