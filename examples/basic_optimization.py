#!/usr/bin/env python3
"""
Basic Optimization Example for diffusion-trt.

This example demonstrates the basic usage of the TensorRT Diffusion
Optimization Pipeline for accelerated image generation on T4 GPUs.

Usage:
    python examples/basic_optimization.py

Requirements:
    - NVIDIA GPU with CUDA support (T4 recommended)
    - diffusion-trt package installed
    - ~10GB free VRAM
"""

import torch
from diffusion_trt.pipeline import PipelineConfig, OptimizedPipeline
from diffusion_trt.utils.vram_monitor import VRAMMonitor


def main():
    """Run basic optimization example."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        print("If running on Google Colab, make sure to enable GPU runtime.")
        return
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Configure the pipeline
    config = PipelineConfig(
        model_id="stabilityai/sdxl-turbo",  # Fast 4-step model
        enable_int8=True,                    # Enable INT8 quantization
        enable_caching=True,                 # Enable feature caching
        num_inference_steps=4,               # SDXL-Turbo uses 4 steps
        guidance_scale=0.0,                  # No CFG for SDXL-Turbo
        seed=42,                             # For reproducible results
    )
    
    print("Loading and optimizing model...")
    print("This may take a few minutes on first run (downloading + compiling)")
    print()
    
    # Load and optimize the pipeline
    # This performs: Load → Calibrate → Quantize → Compile → Setup Caching
    with VRAMMonitor() as monitor:
        pipeline = OptimizedPipeline.from_pretrained(
            config.model_id,
            config=config,
        )
    
    print(f"Model loaded and optimized!")
    print(f"Peak VRAM during loading: {monitor.peak_gb:.2f} GB")
    print()
    
    # Generate an image
    prompt = "A beautiful sunset over a calm ocean, vibrant colors, photorealistic"
    
    print(f"Generating image with prompt: '{prompt}'")
    print()
    
    with VRAMMonitor() as monitor:
        images = pipeline(prompt)
    
    print(f"Image generated!")
    print(f"Peak VRAM during inference: {monitor.peak_gb:.2f} GB")
    print()
    
    # Save the image
    output_path = "output_basic.png"
    images[0].save(output_path)
    print(f"Image saved to: {output_path}")
    print()
    
    # Run a quick benchmark
    print("Running benchmark (5 iterations)...")
    metrics = pipeline.benchmark(
        prompt=prompt,
        num_iterations=5,
        warmup_iterations=2,
    )
    
    print(f"\nBenchmark Results:")
    print(f"  Latency: {metrics.latency_mean_ms:.0f}ms ± {metrics.latency_std_ms:.0f}ms")
    print(f"  Throughput: {metrics.throughput_images_per_sec:.2f} images/sec")
    print(f"  Peak VRAM: {metrics.vram_peak_gb:.2f} GB")
    print(f"  Cache Hit Rate: {metrics.cache_hit_rate:.1%}")


if __name__ == "__main__":
    main()
