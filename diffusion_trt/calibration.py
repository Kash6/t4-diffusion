"""
Calibration engine for INT8 Post-Training Quantization.

This module provides calibration data generation for INT8 PTQ using representative
prompts. It generates diverse calibration batches with latents, timesteps, and
encoder hidden states for optimal quantization accuracy.

Requirements covered:
- 2.1: Generate at least 100 samples for calibration
- 2.2: Use pipeline's text encoder and tokenizer for encoding prompts
- 2.3: Include latents, timesteps, and encoder hidden states in calibration batches
- 2.4: Support streaming iteration to minimize memory footprint
- 2.5: Run UNet in eval mode with no_grad context when collecting activations
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


# Default calibration prompts covering various domains for diverse activation ranges
DEFAULT_CALIBRATION_PROMPTS: List[str] = [
    # Portraits and people
    "A professional headshot photo of a business executive",
    "A candid street photography portrait of an elderly person",
    "A fashion model in dramatic studio lighting",
    "A child playing in a sunny park",
    
    # Landscapes and nature
    "A beautiful sunset over mountains with orange and purple sky",
    "A serene lake reflecting autumn trees",
    "A dramatic thunderstorm over open plains",
    "A tropical beach with crystal clear water",
    
    # Architecture and urban
    "Modern minimalist architecture with clean lines",
    "A bustling city street at night with neon lights",
    "Ancient ruins covered in moss and vines",
    "A cozy coffee shop interior with warm lighting",
    
    # Art styles
    "An oil painting in the style of impressionism",
    "A digital art concept of a futuristic city",
    "A watercolor illustration of flowers",
    "Abstract geometric patterns in vibrant colors",
    
    # Objects and products
    "Product photography of a luxury watch on marble",
    "A vintage camera on a wooden desk",
    "Fresh fruits arranged on a kitchen counter",
    "A sports car in a dramatic studio setting",
    
    # Fantasy and sci-fi
    "A magical forest with glowing mushrooms",
    "A spaceship orbiting a distant planet",
    "A dragon flying over a medieval castle",
    "An underwater city with bioluminescent creatures",
    
    # Animals
    "A majestic lion in the African savanna",
    "A colorful parrot in a tropical rainforest",
    "A playful puppy running through grass",
    "A whale breaching in the ocean at sunset",
    
    # Food and culinary
    "A gourmet dish plated elegantly",
    "Fresh baked bread with steam rising",
    "A colorful smoothie bowl with toppings",
    "Italian pasta with fresh tomatoes and basil",
]


@dataclass
class CalibrationConfig:
    """
    Configuration for calibration data generation.
    
    Attributes:
        num_samples: Number of calibration samples to generate (minimum 100)
        batch_size: Batch size for calibration data (default 1 for T4 VRAM)
        image_size: Target image size as (height, width) tuple
        num_inference_steps: Number of diffusion steps (4 for SDXL-Turbo, 20 for SD 1.5)
        seed: Random seed for reproducibility (None for random)
        latent_channels: Number of channels in latent space (4 for SD, 4 for SDXL)
        latent_scale_factor: Scale factor for latent dimensions (8 for most models)
    """
    num_samples: int = 512
    batch_size: int = 1
    image_size: Tuple[int, int] = (512, 512)
    num_inference_steps: int = 4  # For SDXL-Turbo
    seed: Optional[int] = None
    latent_channels: int = 4
    latent_scale_factor: int = 8
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Requirement 2.1: Generate at least 100 samples
        if self.num_samples < 100:
            raise ValueError(
                f"num_samples must be at least 100 for adequate calibration, "
                f"got {self.num_samples}"
            )
        
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be at least 1, got {self.batch_size}"
            )
        
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError(
                f"image_size dimensions must be positive, got {self.image_size}"
            )
        
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be at least 1, got {self.num_inference_steps}"
            )
    
    @property
    def latent_size(self) -> Tuple[int, int]:
        """Calculate latent space dimensions from image size."""
        return (
            self.image_size[0] // self.latent_scale_factor,
            self.image_size[1] // self.latent_scale_factor,
        )


class CalibrationEngine:
    """
    Engine for generating calibration data for INT8 quantization.
    
    Generates diverse calibration batches with latents, timesteps, and encoder
    hidden states. Supports streaming iteration to minimize memory footprint.
    
    Example:
        >>> config = CalibrationConfig(num_samples=128)
        >>> engine = CalibrationEngine(config)
        >>> for batch in engine.create_dataset(prompts, text_encoder, tokenizer):
        ...     # Use batch for calibration
        ...     pass
    """
    
    def __init__(self, config: CalibrationConfig):
        """
        Initialize the CalibrationEngine.
        
        Args:
            config: Calibration configuration
        """
        self.config = config
        self._generator: Optional[torch.Generator] = None
        
        # Initialize random generator if seed is provided
        if config.seed is not None:
            self._generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            self._generator.manual_seed(config.seed)
    
    def create_dataset(
        self,
        prompts: List[str],
        text_encoder: nn.Module,
        tokenizer,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Generate calibration data batches as a streaming iterator.
        
        Creates calibration samples with latents, timesteps, and encoder hidden
        states. Uses streaming iteration to minimize memory footprint.
        
        Args:
            prompts: List of text prompts for calibration. If fewer than num_samples,
                    prompts will be cycled.
            text_encoder: Text encoder module from the diffusion pipeline
            tokenizer: Tokenizer for encoding text prompts
            
        Yields:
            Dictionary containing:
            - "sample": Random latent tensor [batch_size, channels, height, width]
            - "timestep": Sampled timestep tensor [batch_size]
            - "encoder_hidden_states": Encoded prompt embeddings
            
        Requirements:
            - 2.2: Uses pipeline's text encoder and tokenizer
            - 2.3: Includes latents, timesteps, and encoder hidden states
            - 2.4: Supports streaming iteration
        """
        # Extend prompts if needed to reach num_samples
        if len(prompts) < self.config.num_samples:
            # Cycle through prompts to reach desired sample count
            extended_prompts = []
            for i in range(self.config.num_samples):
                extended_prompts.append(prompts[i % len(prompts)])
            prompts = extended_prompts
        else:
            prompts = prompts[:self.config.num_samples]
        
        # Determine device from text encoder
        device = next(text_encoder.parameters()).device
        
        # Set text encoder to eval mode
        text_encoder.eval()
        
        # Generate samples in batches (streaming to minimize memory)
        num_batches = (self.config.num_samples + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, self.config.num_samples)
            batch_prompts = prompts[start_idx:end_idx]
            actual_batch_size = len(batch_prompts)
            
            # Generate random latents
            latent_height, latent_width = self.config.latent_size
            latents = torch.randn(
                actual_batch_size,
                self.config.latent_channels,
                latent_height,
                latent_width,
                device=device,
                dtype=torch.float16,
                generator=self._generator,
            )
            
            # Sample random timesteps uniformly across the diffusion schedule
            timesteps = torch.randint(
                0,
                1000,  # Standard diffusion schedule has 1000 timesteps
                (actual_batch_size,),
                device=device,
                dtype=torch.long,
            )
            
            # Encode prompts to get encoder hidden states
            # Requirement 2.2: Use pipeline's text encoder and tokenizer
            with torch.no_grad():
                encoder_hidden_states = self._encode_prompts(
                    batch_prompts,
                    text_encoder,
                    tokenizer,
                    device,
                )
            
            # Yield batch as dictionary
            # Requirement 2.3: Include latents, timesteps, and encoder hidden states
            yield {
                "sample": latents,
                "timestep": timesteps,
                "encoder_hidden_states": encoder_hidden_states,
            }
    
    def _encode_prompts(
        self,
        prompts: List[str],
        text_encoder: nn.Module,
        tokenizer,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode text prompts to embeddings using the text encoder.
        
        Args:
            prompts: List of text prompts to encode
            text_encoder: Text encoder module
            tokenizer: Tokenizer for text processing
            device: Target device for tensors
            
        Returns:
            Encoded prompt embeddings tensor
        """
        # Tokenize prompts
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = text_inputs.input_ids.to(device)
        
        # Get encoder hidden states
        with torch.no_grad():
            encoder_output = text_encoder(input_ids)
            
            # Handle different text encoder output formats
            if hasattr(encoder_output, "last_hidden_state"):
                encoder_hidden_states = encoder_output.last_hidden_state
            elif isinstance(encoder_output, tuple):
                encoder_hidden_states = encoder_output[0]
            else:
                encoder_hidden_states = encoder_output
        
        return encoder_hidden_states.to(dtype=torch.float16)
    
    def collect_activations(
        self,
        unet: nn.Module,
        calibration_data: Iterator[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collect activation statistics from UNet for quantization calibration.
        
        Runs the UNet in eval mode with no_grad context to collect activation
        ranges needed for INT8 scale computation.
        
        Args:
            unet: UNet module from the diffusion pipeline
            calibration_data: Iterator of calibration batches from create_dataset()
            
        Returns:
            Dictionary mapping layer names to activation statistics tensors
            containing min/max values for each layer.
            
        Requirements:
            - 2.5: Run UNet in eval mode with no_grad context
        """
        # Requirement 2.5: Set UNet to eval mode
        unet.eval()
        
        # Storage for activation statistics
        activation_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Register hooks to collect activation statistics
        hooks = []
        
        def create_hook(name: str):
            def hook(module, input, output):
                # Handle different output types
                if isinstance(output, torch.Tensor):
                    tensor = output
                elif isinstance(output, tuple) and len(output) > 0:
                    tensor = output[0] if isinstance(output[0], torch.Tensor) else None
                else:
                    return
                
                if tensor is None:
                    return
                
                # Compute min/max for this activation
                with torch.no_grad():
                    batch_min = tensor.min().detach()
                    batch_max = tensor.max().detach()
                
                # Update running statistics
                if name not in activation_stats:
                    activation_stats[name] = {
                        "min": batch_min,
                        "max": batch_max,
                        "count": 1,
                    }
                else:
                    activation_stats[name]["min"] = torch.min(
                        activation_stats[name]["min"], batch_min
                    )
                    activation_stats[name]["max"] = torch.max(
                        activation_stats[name]["max"], batch_max
                    )
                    activation_stats[name]["count"] += 1
            
            return hook
        
        # Register hooks on all modules
        for name, module in unet.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        try:
            # Requirement 2.5: Run with no_grad context
            with torch.no_grad():
                for batch in calibration_data:
                    sample = batch["sample"]
                    timestep = batch["timestep"]
                    encoder_hidden_states = batch["encoder_hidden_states"]
                    
                    # Forward pass through UNet
                    _ = unet(
                        sample,
                        timestep,
                        encoder_hidden_states=encoder_hidden_states,
                    )
        finally:
            # Remove all hooks
            for hook in hooks:
                hook.remove()
        
        # Convert statistics to final format
        result: Dict[str, torch.Tensor] = {}
        for name, stats in activation_stats.items():
            result[f"{name}_min"] = stats["min"]
            result[f"{name}_max"] = stats["max"]
        
        return result
    
    def get_default_prompts(self) -> List[str]:
        """
        Get default calibration prompts covering various domains.
        
        Returns:
            List of diverse prompts for calibration
        """
        return DEFAULT_CALIBRATION_PROMPTS.copy()
    
    def generate_random_prompts(self, num_prompts: int) -> List[str]:
        """
        Generate random prompts by combining templates with subjects.
        
        Useful for creating additional diverse prompts beyond the defaults.
        
        Args:
            num_prompts: Number of prompts to generate
            
        Returns:
            List of generated prompts
        """
        templates = [
            "A photo of {}",
            "A painting of {}",
            "A digital art of {}",
            "A realistic image of {}",
            "An artistic rendering of {}",
            "A detailed illustration of {}",
        ]
        
        subjects = [
            "a mountain landscape",
            "a city skyline",
            "a forest path",
            "an ocean view",
            "a desert scene",
            "a snowy village",
            "a tropical island",
            "a medieval castle",
            "a modern building",
            "a vintage car",
            "a cute animal",
            "a beautiful flower",
            "a starry night sky",
            "a rainy street",
            "a sunny beach",
        ]
        
        prompts = []
        for i in range(num_prompts):
            template = templates[i % len(templates)]
            subject = subjects[i % len(subjects)]
            prompts.append(template.format(subject))
        
        return prompts
