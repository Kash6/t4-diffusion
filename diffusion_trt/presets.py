"""
Fixed-resolution engine presets for optimal T4 performance.

Static TensorRT engines are optimized for specific resolutions and use
less VRAM than dynamic engines while providing better performance.

Supported presets:
- SD15_512: SD 1.5 at 512×512
- SDXL_512: SDXL/SDXL-Turbo at 512×512
- SDXL_768: SDXL at 768×768
- SDXL_1024: SDXL at 1024×1024 (may exceed T4 VRAM with INT8)

Usage:
    from diffusion_trt.presets import get_preset, PRESETS
    
    config = get_preset("SDXL_512")
    pipeline = OptimizedPipeline.from_pretrained(config.model_id, config=config)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .pipeline import PipelineConfig


@dataclass
class EnginePreset:
    """
    Preset configuration for a fixed-resolution TensorRT engine.
    
    Attributes:
        name: Human-readable preset name
        model_id: HuggingFace model identifier
        image_size: Fixed (height, width) for this preset
        num_inference_steps: Recommended steps for this model
        guidance_scale: Recommended CFG scale
        estimated_vram_gb: Estimated peak VRAM usage
        description: Brief description of the preset
    """
    name: str
    model_id: str
    image_size: Tuple[int, int]
    num_inference_steps: int
    guidance_scale: float
    estimated_vram_gb: float
    description: str


# Available presets
PRESETS: Dict[str, EnginePreset] = {
    "SD15_512": EnginePreset(
        name="SD 1.5 @ 512×512",
        model_id="runwayml/stable-diffusion-v1-5",
        image_size=(512, 512),
        num_inference_steps=20,
        guidance_scale=7.5,
        estimated_vram_gb=6.0,
        description="Stable Diffusion 1.5 at standard 512×512 resolution. "
                    "Good balance of quality and speed.",
    ),
    "SD15_768": EnginePreset(
        name="SD 1.5 @ 768×768",
        model_id="runwayml/stable-diffusion-v1-5",
        image_size=(768, 768),
        num_inference_steps=20,
        guidance_scale=7.5,
        estimated_vram_gb=8.5,
        description="Stable Diffusion 1.5 at higher 768×768 resolution. "
                    "Better detail but slower.",
    ),
    "SDXL_TURBO_512": EnginePreset(
        name="SDXL-Turbo @ 512×512",
        model_id="stabilityai/sdxl-turbo",
        image_size=(512, 512),
        num_inference_steps=4,
        guidance_scale=0.0,
        estimated_vram_gb=8.0,
        description="SDXL-Turbo at 512×512. Ultra-fast 4-step generation. "
                    "Best for real-time applications.",
    ),
    "SDXL_768": EnginePreset(
        name="SDXL @ 768×768",
        model_id="stabilityai/sdxl-turbo",
        image_size=(768, 768),
        num_inference_steps=4,
        guidance_scale=0.0,
        estimated_vram_gb=10.5,
        description="SDXL-Turbo at 768×768. Higher quality output. "
                    "Recommended for T4 with INT8.",
    ),
    "SDXL_1024": EnginePreset(
        name="SDXL @ 1024×1024",
        model_id="stabilityai/sdxl-turbo",
        image_size=(1024, 1024),
        num_inference_steps=4,
        guidance_scale=0.0,
        estimated_vram_gb=14.0,
        description="SDXL-Turbo at native 1024×1024. Highest quality but "
                    "may approach T4 VRAM limit. Use with caution.",
    ),
}


def get_preset(preset_name: str) -> PipelineConfig:
    """
    Get a PipelineConfig for the specified preset.
    
    Args:
        preset_name: Name of the preset (e.g., "SDXL_512", "SD15_768")
        
    Returns:
        PipelineConfig configured for the preset
        
    Raises:
        ValueError: If preset_name is not recognized
        
    Example:
        >>> config = get_preset("SDXL_TURBO_512")
        >>> pipeline = OptimizedPipeline.from_pretrained(config.model_id, config=config)
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset: '{preset_name}'. "
            f"Available presets: {available}"
        )
    
    preset = PRESETS[preset_name]
    
    return PipelineConfig(
        model_id=preset.model_id,
        enable_int8=True,
        enable_caching=True,
        num_inference_steps=preset.num_inference_steps,
        guidance_scale=preset.guidance_scale,
        image_size=preset.image_size,
    )


def list_presets() -> Dict[str, dict]:
    """
    List all available presets with their details.
    
    Returns:
        Dictionary mapping preset names to their details
        
    Example:
        >>> for name, info in list_presets().items():
        ...     print(f"{name}: {info['description']}")
    """
    return {
        name: {
            "name": preset.name,
            "model_id": preset.model_id,
            "image_size": preset.image_size,
            "steps": preset.num_inference_steps,
            "estimated_vram_gb": preset.estimated_vram_gb,
            "description": preset.description,
        }
        for name, preset in PRESETS.items()
    }


def get_recommended_preset(
    max_vram_gb: float = 15.6,
    prefer_quality: bool = False,
) -> str:
    """
    Get the recommended preset based on available VRAM and preferences.
    
    Args:
        max_vram_gb: Maximum available VRAM in GB (default: T4's 15.6GB)
        prefer_quality: If True, prefer higher resolution over speed
        
    Returns:
        Name of the recommended preset
        
    Example:
        >>> preset_name = get_recommended_preset(max_vram_gb=12.0)
        >>> config = get_preset(preset_name)
    """
    # Filter presets that fit in available VRAM (with 2GB safety margin)
    available_vram = max_vram_gb - 2.0
    
    valid_presets = [
        (name, preset)
        for name, preset in PRESETS.items()
        if preset.estimated_vram_gb <= available_vram
    ]
    
    if not valid_presets:
        # Fall back to smallest preset
        return "SD15_512"
    
    if prefer_quality:
        # Sort by resolution (descending), then by VRAM (ascending)
        valid_presets.sort(
            key=lambda x: (
                -x[1].image_size[0] * x[1].image_size[1],
                x[1].estimated_vram_gb
            )
        )
    else:
        # Sort by speed (fewer steps first), then by VRAM (ascending)
        valid_presets.sort(
            key=lambda x: (
                x[1].num_inference_steps,
                x[1].estimated_vram_gb
            )
        )
    
    return valid_presets[0][0]


# Convenience exports
__all__ = [
    "EnginePreset",
    "PRESETS",
    "get_preset",
    "list_presets",
    "get_recommended_preset",
]
