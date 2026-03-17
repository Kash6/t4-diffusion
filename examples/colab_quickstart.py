#!/usr/bin/env python3
"""
Google Colab Quickstart for diffusion-trt.

This script is designed to run in Google Colab's free tier with T4 GPU.
It demonstrates the complete workflow from installation to image generation.

To use in Colab:
1. Create a new notebook
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Copy and paste each cell below

Cell 1: Installation
---
!pip install git+https://github.com/Kash6/TensorRTDiffusionMO.git
!pip install hypothesis pytest  # For running tests

Cell 2: Verify GPU
---
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

Cell 3: Run this script
---
!python -c "exec(open('examples/colab_quickstart.py').read())"
"""

import torch
from diffusion_trt.pipeline import PipelineConfig, OptimizedPipeline
from diffusion_trt.utils.vram_monitor import VRAMMonitor, get_vram_usage


def check_environment():
    """Check that we're running on a compatible GPU."""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   Please enable GPU runtime in Colab:")
        print("   Runtime → Change runtime type → T4 GPU")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✓ GPU detected: {gpu_name}")
    print(f"✓ Total VRAM: {total_vram:.1f} GB")
    
    if "T4" in gpu_name:
        print("✓ Running on T4 GPU (optimal for this package)")
    elif total_vram < 12:
        print("⚠ Warning: Less than 12GB VRAM. Some models may not fit.")
    
    return True


def generate_images():
    """Generate images using the optimized pipeline."""
    
    print("\n" + "="*60)
    print("STEP 1: Configure Pipeline")
    print("="*60)
    
    config = PipelineConfig(
        model_id="stabilityai/sdxl-turbo",
        enable_int8=True,
        enable_caching=True,
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=42,
    )
    
    print(f"  Model: {config.model_id}")
    print(f"  INT8 Quantization: {config.enable_int8}")
    print(f"  Feature Caching: {config.enable_caching}")
    print(f"  Inference Steps: {config.num_inference_steps}")
    
    print("\n" + "="*60)
    print("STEP 2: Load and Optimize Model")
    print("="*60)
    print("  This may take 2-5 minutes on first run...")
    
    with VRAMMonitor() as monitor:
        pipeline = OptimizedPipeline.from_pretrained(
            config.model_id,
            config=config,
        )
    
    print(f"  ✓ Model loaded and optimized!")
    print(f"  ✓ Peak VRAM: {monitor.peak_gb:.2f} GB")
    
    print("\n" + "="*60)
    print("STEP 3: Generate Images")
    print("="*60)
    
    prompts = [
        "A serene Japanese garden with cherry blossoms and a koi pond",
        "A futuristic cityscape at night with neon lights",
        "A cozy cabin in the mountains during winter",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n  Generating image {i}/{len(prompts)}...")
        print(f"  Prompt: '{prompt[:50]}...'")
        
        with VRAMMonitor() as monitor:
            images = pipeline(prompt)
        
        output_path = f"output_{i}.png"
        images[0].save(output_path)
        
        print(f"  ✓ Saved to: {output_path}")
        print(f"  ✓ VRAM used: {monitor.peak_gb:.2f} GB")
    
    print("\n" + "="*60)
    print("STEP 4: Run Benchmark")
    print("="*60)
    
    metrics = pipeline.benchmark(
        prompt="A beautiful landscape",
        num_iterations=5,
        warmup_iterations=2,
    )
    
    print(f"\n  Benchmark Results:")
    print(f"  ├─ Latency: {metrics.latency_mean_ms:.0f}ms ± {metrics.latency_std_ms:.0f}ms")
    print(f"  ├─ Throughput: {metrics.throughput_images_per_sec:.2f} img/s")
    print(f"  ├─ Peak VRAM: {metrics.vram_peak_gb:.2f} GB")
    print(f"  └─ Cache Hit Rate: {metrics.cache_hit_rate:.1%}")
    
    print("\n" + "="*60)
    print("STEP 5: Save Engine for Faster Loading")
    print("="*60)
    
    engine_path = "optimized_engine.pt"
    pipeline.save_engine(engine_path)
    print(f"  ✓ Engine saved to: {engine_path}")
    print(f"  ✓ Next time, use OptimizedPipeline.load_engine('{engine_path}')")
    print(f"    to skip the optimization step!")
    
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print("\nYour images have been generated and saved.")
    print("Check the file browser on the left to view them.")


def main():
    """Main entry point for Colab quickstart."""
    print("="*60)
    print("diffusion-trt: TensorRT Optimization for Stable Diffusion")
    print("Optimized for Google Colab Free Tier (T4 GPU)")
    print("="*60)
    
    if not check_environment():
        return
    
    generate_images()


if __name__ == "__main__":
    main()
