"""
Standalone local test script to debug the INT8 quantization -> ONNX export ->
TensorRT engine build pipeline on a local GPU (e.g., RTX 5080).

This exercises the exact same code path as the Colab notebook but with full
Python tracebacks instead of opaque Colab session crashes.

Usage:
    .venv/bin/python scripts/local_trt_test.py
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

import torch
from diffusion_trt.pipeline import PipelineConfig, OptimizedPipeline


def main():
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config = PipelineConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        enable_int8=True,
        backend="tensorrt",  # opt-in backend (default is now "quanto")
        enable_caching=False,  # isolate the TRT path, no caching noise
        num_inference_steps=20,
        guidance_scale=7.5,
        num_calibration_samples=32,
        seed=42,
    )

    print("\n=== Loading and optimizing model ===")
    pipeline = OptimizedPipeline.from_pretrained(config.model_id, config=config)

    print("\n=== Model loaded ===")
    print(f"is_optimized: {pipeline.is_optimized}")
    print(f"Peak VRAM: {pipeline.get_vram_usage():.2f} GB")

    print("\n=== Running inference ===")
    # Disable safety checker's black-image override so we can visually verify
    # actual pipeline output (this is a local debug script, not production).
    if hasattr(pipeline._pipeline, "safety_checker"):
        pipeline._pipeline.safety_checker = None
    images = pipeline("a scenic mountain landscape at sunset, photorealistic")
    images[0].save("/tmp/local_trt_test_output.png")
    print("Saved output image to /tmp/local_trt_test_output.png")

    print("\n=== Benchmark ===")
    metrics = pipeline.benchmark(
        prompt="a scenic mountain landscape at sunset",
        num_iterations=5,
        warmup_iterations=2,
    )
    print(f"Latency (mean): {metrics.latency_mean_ms:.0f} ms")
    print(f"Throughput: {metrics.throughput_images_per_sec:.2f} img/s")
    print(f"Peak VRAM: {metrics.vram_peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
