#!/usr/bin/env python3
"""
T4 Benchmark Suite for t4-diffusion.

This script runs comprehensive benchmarks on T4 GPU to measure:
- Latency (mean, p50, p95, p99)
- Throughput (images/second)
- VRAM usage (peak, allocated)
- Speedup vs FP16 baseline

Configurations tested:
- SD 1.5 at 512×512 with 20/50 steps
- SDXL-Turbo at 512×512 with 4 steps
- SDXL base at 768×768 and 1024×1024 with 20 steps

Usage:
    python benchmarks/t4_benchmark_suite.py [--output results.json]
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from diffusion_trt.pipeline import PipelineConfig, OptimizedPipeline
from diffusion_trt.models import BenchmarkMetrics, T4_VRAM_LIMIT_GB
from diffusion_trt.utils.vram_monitor import VRAMMonitor, clear_cache


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    model_id: str
    image_size: tuple
    num_inference_steps: int
    guidance_scale: float
    enable_int8: bool
    enable_caching: bool
    num_iterations: int = 5
    warmup_iterations: int = 2


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    config_name: str
    model_id: str
    image_size: tuple
    num_inference_steps: int
    enable_int8: bool
    enable_caching: bool
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_images_per_sec: float
    vram_peak_gb: float
    vram_allocated_gb: float
    cache_hit_rate: float
    timestamp: str


# Benchmark configurations
BENCHMARK_CONFIGS = [
    # SD 1.5 benchmarks
    BenchmarkConfig(
        name="SD1.5-512-20steps-FP16",
        model_id="runwayml/stable-diffusion-v1-5",
        image_size=(512, 512),
        num_inference_steps=20,
        guidance_scale=7.5,
        enable_int8=False,
        enable_caching=False,
    ),
    BenchmarkConfig(
        name="SD1.5-512-20steps-INT8",
        model_id="runwayml/stable-diffusion-v1-5",
        image_size=(512, 512),
        num_inference_steps=20,
        guidance_scale=7.5,
        enable_int8=True,
        enable_caching=True,
    ),
    BenchmarkConfig(
        name="SD1.5-512-50steps-FP16",
        model_id="runwayml/stable-diffusion-v1-5",
        image_size=(512, 512),
        num_inference_steps=50,
        guidance_scale=7.5,
        enable_int8=False,
        enable_caching=False,
    ),
    BenchmarkConfig(
        name="SD1.5-512-50steps-INT8",
        model_id="runwayml/stable-diffusion-v1-5",
        image_size=(512, 512),
        num_inference_steps=50,
        guidance_scale=7.5,
        enable_int8=True,
        enable_caching=True,
    ),
    
    # SDXL-Turbo benchmarks
    BenchmarkConfig(
        name="SDXL-Turbo-512-4steps-FP16",
        model_id="stabilityai/sdxl-turbo",
        image_size=(512, 512),
        num_inference_steps=4,
        guidance_scale=0.0,
        enable_int8=False,
        enable_caching=False,
    ),
    BenchmarkConfig(
        name="SDXL-Turbo-512-4steps-INT8",
        model_id="stabilityai/sdxl-turbo",
        image_size=(512, 512),
        num_inference_steps=4,
        guidance_scale=0.0,
        enable_int8=True,
        enable_caching=True,
    ),
]


def run_benchmark(config: BenchmarkConfig) -> Optional[BenchmarkResult]:
    """Run a single benchmark configuration."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"{'='*60}")
    print(f"  Model: {config.model_id}")
    print(f"  Size: {config.image_size}")
    print(f"  Steps: {config.num_inference_steps}")
    print(f"  INT8: {config.enable_int8}")
    print(f"  Caching: {config.enable_caching}")
    
    try:
        # Create pipeline config
        pipeline_config = PipelineConfig(
            model_id=config.model_id,
            enable_int8=config.enable_int8,
            enable_caching=config.enable_caching,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            image_size=config.image_size,
            seed=42,
        )
        
        # Load and optimize pipeline
        print("\n  Loading model...")
        start_load = time.time()
        
        pipeline = OptimizedPipeline.from_pretrained(
            config.model_id,
            config=pipeline_config,
        )
        
        load_time = time.time() - start_load
        print(f"  Loaded in {load_time:.1f}s")
        
        # Run benchmark
        print(f"\n  Running benchmark ({config.warmup_iterations} warmup + {config.num_iterations} measured)...")
        
        metrics = pipeline.benchmark(
            prompt="A beautiful landscape with mountains and a lake, photorealistic",
            num_iterations=config.num_iterations,
            warmup_iterations=config.warmup_iterations,
        )
        
        # Create result
        result = BenchmarkResult(
            config_name=config.name,
            model_id=config.model_id,
            image_size=config.image_size,
            num_inference_steps=config.num_inference_steps,
            enable_int8=config.enable_int8,
            enable_caching=config.enable_caching,
            latency_mean_ms=metrics.latency_mean_ms,
            latency_std_ms=metrics.latency_std_ms,
            latency_p50_ms=metrics.latency_p50_ms,
            latency_p95_ms=metrics.latency_p95_ms,
            latency_p99_ms=metrics.latency_p99_ms,
            throughput_images_per_sec=metrics.throughput_images_per_sec,
            vram_peak_gb=metrics.vram_peak_gb,
            vram_allocated_gb=metrics.vram_allocated_gb,
            cache_hit_rate=metrics.cache_hit_rate,
            timestamp=datetime.now().isoformat(),
        )
        
        # Print results
        print(f"\n  Results:")
        print(f"    Latency: {metrics.latency_mean_ms:.0f}ms ± {metrics.latency_std_ms:.0f}ms")
        print(f"    P50/P95/P99: {metrics.latency_p50_ms:.0f}/{metrics.latency_p95_ms:.0f}/{metrics.latency_p99_ms:.0f}ms")
        print(f"    Throughput: {metrics.throughput_images_per_sec:.2f} img/s")
        print(f"    VRAM Peak: {metrics.vram_peak_gb:.2f} GB")
        
        # Cleanup
        del pipeline
        clear_cache()
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        clear_cache()
        gc.collect()
        return None


def calculate_speedups(results: List[BenchmarkResult]) -> Dict[str, float]:
    """Calculate speedups between baseline and optimized configurations."""
    speedups = {}
    
    # Group results by model and settings
    baselines = {r.config_name: r for r in results if not r.enable_int8}
    optimized = {r.config_name: r for r in results if r.enable_int8}
    
    # Calculate speedups
    pairs = [
        ("SD1.5-512-20steps-FP16", "SD1.5-512-20steps-INT8"),
        ("SD1.5-512-50steps-FP16", "SD1.5-512-50steps-INT8"),
        ("SDXL-Turbo-512-4steps-FP16", "SDXL-Turbo-512-4steps-INT8"),
    ]
    
    for baseline_name, opt_name in pairs:
        if baseline_name in baselines and opt_name.replace("-FP16", "-INT8") in [r.config_name for r in results]:
            baseline = baselines[baseline_name]
            opt = next((r for r in results if r.config_name == opt_name), None)
            if opt:
                speedup = baseline.latency_mean_ms / opt.latency_mean_ms
                speedups[f"{baseline_name} → {opt_name}"] = speedup
    
    return speedups


def generate_report(results: List[BenchmarkResult], speedups: Dict[str, float]) -> str:
    """Generate markdown benchmark report."""
    report = []
    report.append("# T4 Benchmark Results")
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append("| Configuration | Latency (ms) | Throughput (img/s) | VRAM (GB) |")
    report.append("|--------------|-------------|-------------------|-----------|")
    
    for r in results:
        report.append(
            f"| {r.config_name} | {r.latency_mean_ms:.0f} ± {r.latency_std_ms:.0f} | "
            f"{r.throughput_images_per_sec:.2f} | {r.vram_peak_gb:.2f} |"
        )
    
    report.append("")
    report.append("## Speedups")
    report.append("")
    
    for name, speedup in speedups.items():
        status = "✓" if speedup >= 1.5 else "○"
        report.append(f"- {status} {name}: **{speedup:.2f}x**")
    
    report.append("")
    report.append("## Notes")
    report.append("")
    report.append("- Speedup target: >= 1.5x (marked with ✓)")
    report.append("- VRAM limit: 15.6 GB (T4)")
    report.append("- All benchmarks use seed=42 for reproducibility")
    report.append("- Latency includes full pipeline (encode → denoise → decode)")
    
    return "\n".join(report)


def main():
    """Run the T4 benchmark suite."""
    parser = argparse.ArgumentParser(description="T4 Benchmark Suite")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for results (JSON)")
    parser.add_argument("--report", type=str, default="BENCHMARK_REPORT.md",
                        help="Output file for markdown report")
    args = parser.parse_args()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("="*60)
    print("T4 BENCHMARK SUITE")
    print("="*60)
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram:.1f} GB")
    print(f"T4 Limit: {T4_VRAM_LIMIT_GB} GB")
    print(f"Configs to run: {len(BENCHMARK_CONFIGS)}")
    
    # Run benchmarks
    results = []
    for config in BENCHMARK_CONFIGS:
        result = run_benchmark(config)
        if result:
            results.append(result)
    
    # Calculate speedups
    speedups = calculate_speedups(results)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, speedup in speedups.items():
        print(f"  {name}: {speedup:.2f}x")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({
            "gpu": gpu_name,
            "vram_gb": total_vram,
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(r) for r in results],
            "speedups": speedups,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Generate report
    report = generate_report(results, speedups)
    report_path = Path(args.report)
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
