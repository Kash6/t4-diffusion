"""
Setup script for the TensorRT Diffusion Model Optimization Pipeline.

This package provides TensorRT-based optimization for diffusion models
targeting NVIDIA T4 GPUs on Google Colab's free tier.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    try:
        long_description = fh.read()
    except FileNotFoundError:
        long_description = "TensorRT Diffusion Model Optimization Pipeline"

setup(
    name="diffusion-trt",
    version="0.1.0",
    author="Diffusion TRT Team",
    author_email="",
    description="TensorRT-based optimization pipeline for diffusion models targeting NVIDIA T4 GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.10",
    # NOTE: torch, torchvision, torch-tensorrt are pre-installed on Colab
    # We don't include them here to avoid version conflicts
    install_requires=[
        "diffusers>=0.25.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "hypothesis>=6.0",
            "pytest-benchmark>=4.0",
        ],
        "tensorrt": [
            # Install these manually on Colab if not present
            "nvidia-modelopt>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # CLI entry points can be added here
        ],
    },
)
