# t4-diffusion Roadmap

This document outlines planned features and research directions for the t4-diffusion package.

## Current Release (v0.1.0)

### Supported Models
- **Stable Diffusion 1.5** (`runwayml/stable-diffusion-v1-5`)
- **SDXL-Turbo** (`stabilityai/sdxl-turbo`)

### Supported Resolutions
- 512×512 (SD 1.5, SDXL-Turbo)
- 768×768 (SD 1.5, SDXL-Turbo)
- 1024×1024 (SDXL-Turbo - use with caution on T4)

### Features
- INT8 Post-Training Quantization via TensorRT Model Optimizer
- DeepCache-style feature caching
- VRAM monitoring with 15.6GB T4 limit enforcement
- Fixed-resolution engine presets for optimal performance
- Property-based testing for correctness guarantees

---

## Planned: LCM/LCM-LoRA Support (v0.2.0)

### Research Summary

**LCM (Latent Consistency Models)** are designed to be perturbation-resistant, making them excellent candidates for INT8 quantization. Key findings:

1. **Quantization Compatibility**: OpenVINO/NNCF has demonstrated successful INT8 PTQ on LCM models with consistent generation results. LCM's training methodology makes it inherently robust to precision reduction.

2. **Official TensorRT Support**: NVIDIA provides official TensorRT engines for SDXL-LCM and SDXL-LCM-LoRA at [stabilityai/stable-diffusion-xl-1.0-tensorrt](https://huggingface.co/stabilityai/stable-diffusion-xl-1.0-tensorrt).

3. **Performance on A100/H100**: Official benchmarks show LCM with TensorRT achieves:
   - A100: 426ms total for 4 steps at 1024×1024
   - H100: 234ms total for 4 steps at 1024×1024
   - T4 benchmarks not published, but expected ~1-2s for 4 steps at 512×512

4. **LCM-LoRA Advantage**: LCM-LoRA is a universal acceleration module that can be applied to any SD/SDXL fine-tune, reducing steps from 25-50 to 2-8 steps.

### Planned Features

#### LCM-LoRA Integration
```python
# Target API
from diffusion_trt import OptimizedPipeline, PipelineConfig

config = PipelineConfig(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    enable_int8=True,
    enable_caching=True,
    use_lcm_lora=True,  # NEW: Auto-apply LCM-LoRA
    num_inference_steps=4,  # LCM optimal: 2-8 steps
    guidance_scale=1.5,  # LCM optimal: ~1.5
)

pipeline = OptimizedPipeline.from_pretrained(config.model_id, config=config)
```

#### Supported LCM-LoRA Variants
- `latent-consistency/lcm-lora-sdxl` - For SDXL base
- `latent-consistency/lcm-lora-sd15` - For SD 1.5 (if available)

#### Target Performance (T4 GPU)
| Configuration | Steps | Target Latency | Resolution |
|--------------|-------|----------------|------------|
| SDXL + LCM-LoRA + INT8 | 4 | <1.5s | 512×512 |
| SDXL + LCM-LoRA + INT8 | 4 | <2.5s | 768×768 |
| SD 1.5 + LCM-LoRA + INT8 | 4 | <0.8s | 512×512 |

### Implementation Tasks

1. **Add LCM-LoRA loader** to ModelLoader
   - Auto-download from HuggingFace
   - Merge LoRA weights with base model
   - Support custom LoRA scale (default: 1.0)

2. **Add LCMScheduler support** to pipeline
   - Detect LCM mode and switch scheduler
   - Adjust default CFG scale to ~1.5
   - Adjust default steps to 4

3. **Validate INT8 + LCM-LoRA combination**
   - Ensure quantization doesn't degrade LCM quality
   - Test with various prompts and seeds
   - Measure VRAM usage on T4

4. **Add LCM presets** to presets.py
   - `SDXL_LCM_512`, `SDXL_LCM_768`
   - `SD15_LCM_512`

---

## Planned: SDXL Lightning Support (v0.3.0)

### Research Summary

**SDXL-Lightning** by ByteDance is a progressive adversarial diffusion distillation model that achieves high-quality 1024px images in 2-8 steps.

Key differences from LCM:
- Uses adversarial training (vs consistency distillation)
- Available as both full checkpoint and LoRA
- Generally produces higher quality at same step count

### Planned Features

- Support for `ByteDance/SDXL-Lightning` LoRA variants (2-step, 4-step, 8-step)
- Automatic scheduler configuration for Lightning
- Comparison benchmarks: LCM vs Lightning on T4

---

## Planned: Hyper-SD Support (v0.4.0)

### Research Summary

**Hyper-SD** by ByteDance combines consistency distillation with adversarial training for even faster generation.

### Planned Features

- Support for Hyper-SD LoRA variants
- 1-step generation capability
- Quality comparison with LCM and Lightning

---

## Not Planned (T4 Limitations)

The following are explicitly NOT planned due to T4 VRAM constraints:

### SD3/SD3.5 Large
- SD3.5 Large requires FP8 optimizations for best performance
- T4 is Turing architecture (no FP8 Tensor Cores)
- VRAM requirements exceed T4's 15.6GB for optimal quality

### FLUX Models
- FLUX.1 models require 24GB+ VRAM for comfortable operation
- Not suitable for free Colab T4 tier

### 1024×1024 with Full Features
- 1024×1024 + INT8 + caching approaches T4 VRAM limit
- Recommend 512×512 or 768×768 for reliable operation

---

## Contributing

We welcome contributions! Priority areas:

1. **LCM-LoRA integration** - Help implement v0.2.0 features
2. **Benchmark data** - Run benchmarks on T4 and share results
3. **Documentation** - Improve examples and tutorials
4. **Testing** - Add more property tests for edge cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Version History

- **v0.1.0** (Current): Initial release with SD 1.5, SDXL-Turbo, INT8, caching
- **v0.2.0** (Planned): LCM-LoRA support
- **v0.3.0** (Planned): SDXL-Lightning support
- **v0.4.0** (Planned): Hyper-SD support
