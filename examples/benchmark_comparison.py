#!/usr/bin/env python3
"""
Benchmark Comparison Example for diffusion-trt.

This example compares the performance of:
1. Baseline FP16 inference (no optimizations)
2. Optimized INT8 + TensorRT + Caching inference

Usage:
    python examples/benchmark_comparison.py

Requirements:
    - NVIDIA GPU with CUDA support (T4 recommended)
    - diffusion-trt package installed
    - ~10GB free VRAM
"""

import gc
import torch
from diffusion_trt.pipeline import PipelineConfig, OptimizedPipeline
from diffusion_trt.utils.vram_monitor import VRAMMonitor, clear_cache


def run_benchmark(pipeline, name: str, prompt: str, num_iterations: int = 5):
    """Run benchmark and print results."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    metrics = pipeline.benchmark(
        prompt=prompt,
        num_iterations=num_iterations,
        warmup_iterations=2,
    )
    
    print(f"  Latency (mean):  {metrics.latency_mean_ms:.0f} ms")
    print(f"  Latency (std):   {metrics.latency_std_ms:.0f} ms")
    print(f"  Latency (p50):   {metrics.latency_p50_ms:.0f} ms")
    print(f"  Latency (p95):   {metrics.latency_p95_ms:.0f} ms")
    print(f"  Throughput:      {metrics.throughput_images_per_sec:.2f} img/s")
    print(f"  Peak VRAM:       {metrics.vram_peak_gb:.2f} GB")
    print(f"  Cache Hit Rate:  {metrics.cache_hit_rate:.1%}")
    
    return metrics


def main():
    """Run benchmark comparison."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    model_id = "stabilityai/sdxl-turbo"
    prompt = "A majestic mountain landscape with snow-capped peaks and a crystal clear lake"
    
    # =========================================================================
    # Baseline: FP16, no INT8, no caching
    # =========================================================================
    print("\n" + "="*60)
    print("Loading BASELINE pipeline (FP16, no optimizations)...")
    print("="*60)
    
    baseline_config = PipelineConfig(
        model_id=model_id,
        enable_int8=False,
        enable_caching=False,
        num_inference_steps=4,
        seed=42,
    )
    
    baseline_pipeline = OptimizedPipeline.from_pretrained(
        model_id,
        config=baseline_config,
    )
    
    baseline_metrics = run_benchmark(
        baseline_pipeline,
        "Baseline (FP16)",
        prompt,
    )
    
    # Clean up baseline to free VRAM
    del baseline_pipeline
    clear_cache()
    gc.collect()
    
    # =========================================================================
    # Optimized: INT8 + TensorRT + Caching
    # =========================================================================
    print("\n" + "="*60)
    print("Loading OPTIMIZED pipeline (INT8 + TensorRT + Caching)...")
    print("="*60)
    
    optimized_config = PipelineConfig(
        model_id=model_id,
        enable_int8=True,
        enable_caching=True,
        num_inference_steps=4,
        seed=42,
    )
    
    optimized_pipeline = OptimizedPipeline.from_pretrained(
        model_id,
        config=optimized_config,
    )
    
    optimized_metrics = run_benchmark(
        optimized_pipeline,
        "Optimized (INT8 + TRT + Cache)",
        prompt,
    )
    
    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    speedup = baseline_metrics.latency_mean_ms / optimized_metrics.latency_mean_ms
    vram_reduction = baseline_metrics.vram_peak_gb - optimized_metrics.vram_peak_gb
    
    print(f"\n  Baseline Latency:   {baseline_metrics.latency_mean_ms:.0f} ms")
    print(f"  Optimized Latency:  {optimized_metrics.latency_mean_ms:.0f} ms")
    print(f"  Speedup:            {speedup:.2f}x")
    print()
    print(f"  Baseline VRAM:      {baseline_metrics.vram_peak_gb:.2f} GB")
    print(f"  Optimized VRAM:     {optimized_metrics.vram_peak_gb:.2f} GB")
    print(f"  VRAM Reduction:     {vram_reduction:.2f} GB")
    print()
    
    if speedup >= 1.5:
        print(f"  ✓ Achieved target speedup (>= 1.5x)")
    else:
        print(f"  ✗ Below target speedup (>= 1.5x)")
    
    if optimized_metrics.vram_peak_gb <= 15.6:
        print(f"  ✓ Within T4 VRAM limit (15.6 GB)")
    else:
        print(f"  ✗ Exceeded T4 VRAM limit (15.6 GB)")


if __name__ == "__main__":
    main()
