"""
Unit tests for CalibrationEngine component in diffusion_trt.calibration.

Tests cover:
- CalibrationConfig validation (num_samples >= 100, batch_size >= 1, etc.)
- CalibrationEngine.create_dataset() with mocked text_encoder and tokenizer
- Streaming iteration yields correct batch structure
- Correct number of samples generated
- CalibrationEngine.collect_activations() with mocked UNet
- Default prompts and random prompt generation

Validates: Requirements 2.1, 2.4
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Iterator, Dict, List

import torch
import torch.nn as nn

from diffusion_trt.calibration import (
    CalibrationConfig,
    CalibrationEngine,
    DEFAULT_CALIBRATION_PROMPTS,
)


# =============================================================================
# CalibrationConfig Tests - Valid Construction
# =============================================================================


class TestCalibrationConfigValidConstruction:
    """Test valid construction of CalibrationConfig."""

    @pytest.mark.unit
    def test_valid_default_config(self):
        """Test creating a CalibrationConfig with default values."""
        config = CalibrationConfig()
        assert config.num_samples == 512
        assert config.batch_size == 1
        assert config.image_size == (512, 512)
        assert config.num_inference_steps == 4
        assert config.seed is None
        assert config.latent_channels == 4
        assert config.latent_scale_factor == 8

    @pytest.mark.unit
    def test_valid_custom_config(self):
        """Test creating a CalibrationConfig with custom values."""
        config = CalibrationConfig(
            num_samples=256,
            batch_size=2,
            image_size=(768, 768),
            num_inference_steps=20,
            seed=42,
            latent_channels=4,
            latent_scale_factor=8,
        )
        assert config.num_samples == 256
        assert config.batch_size == 2
        assert config.image_size == (768, 768)
        assert config.num_inference_steps == 20
        assert config.seed == 42

    @pytest.mark.unit
    def test_valid_minimum_samples(self):
        """Test CalibrationConfig with minimum valid num_samples (100)."""
        config = CalibrationConfig(num_samples=100)
        assert config.num_samples == 100

    @pytest.mark.unit
    def test_valid_large_batch_size(self):
        """Test CalibrationConfig with larger batch size."""
        config = CalibrationConfig(batch_size=4)
        assert config.batch_size == 4

    @pytest.mark.unit
    def test_latent_size_property(self):
        """Test latent_size property calculation."""
        config = CalibrationConfig(
            image_size=(512, 512),
            latent_scale_factor=8,
        )
        assert config.latent_size == (64, 64)

    @pytest.mark.unit
    def test_latent_size_non_square(self):
        """Test latent_size property with non-square image."""
        config = CalibrationConfig(
            image_size=(768, 512),
            latent_scale_factor=8,
        )
        assert config.latent_size == (96, 64)


# =============================================================================
# CalibrationConfig Tests - Validation Errors
# =============================================================================


class TestCalibrationConfigValidation:
    """Test validation errors for invalid CalibrationConfig inputs."""

    @pytest.mark.unit
    def test_invalid_num_samples_below_minimum(self):
        """Test that num_samples < 100 raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be at least 100"):
            CalibrationConfig(num_samples=99)

    @pytest.mark.unit
    def test_invalid_num_samples_zero(self):
        """Test that num_samples = 0 raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be at least 100"):
            CalibrationConfig(num_samples=0)

    @pytest.mark.unit
    def test_invalid_num_samples_negative(self):
        """Test that negative num_samples raises ValueError."""
        with pytest.raises(ValueError, match="num_samples must be at least 100"):
            CalibrationConfig(num_samples=-50)

    @pytest.mark.unit
    def test_invalid_batch_size_zero(self):
        """Test that batch_size = 0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            CalibrationConfig(batch_size=0)

    @pytest.mark.unit
    def test_invalid_batch_size_negative(self):
        """Test that negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            CalibrationConfig(batch_size=-1)

    @pytest.mark.unit
    def test_invalid_image_size_zero_height(self):
        """Test that image_size with zero height raises ValueError."""
        with pytest.raises(ValueError, match="image_size dimensions must be positive"):
            CalibrationConfig(image_size=(0, 512))

    @pytest.mark.unit
    def test_invalid_image_size_zero_width(self):
        """Test that image_size with zero width raises ValueError."""
        with pytest.raises(ValueError, match="image_size dimensions must be positive"):
            CalibrationConfig(image_size=(512, 0))

    @pytest.mark.unit
    def test_invalid_image_size_negative(self):
        """Test that image_size with negative dimensions raises ValueError."""
        with pytest.raises(ValueError, match="image_size dimensions must be positive"):
            CalibrationConfig(image_size=(-512, 512))

    @pytest.mark.unit
    def test_invalid_num_inference_steps_zero(self):
        """Test that num_inference_steps = 0 raises ValueError."""
        with pytest.raises(ValueError, match="num_inference_steps must be at least 1"):
            CalibrationConfig(num_inference_steps=0)

    @pytest.mark.unit
    def test_invalid_num_inference_steps_negative(self):
        """Test that negative num_inference_steps raises ValueError."""
        with pytest.raises(ValueError, match="num_inference_steps must be at least 1"):
            CalibrationConfig(num_inference_steps=-5)


# =============================================================================
# CalibrationEngine Tests - Initialization
# =============================================================================


class TestCalibrationEngineInit:
    """Test CalibrationEngine initialization."""

    @pytest.mark.unit
    def test_init_with_default_config(self):
        """Test CalibrationEngine initialization with default config."""
        config = CalibrationConfig()
        engine = CalibrationEngine(config)
        assert engine.config == config
        assert engine._generator is None

    @pytest.mark.unit
    def test_init_with_seed(self):
        """Test CalibrationEngine initialization with seed creates generator."""
        config = CalibrationConfig(seed=42)
        engine = CalibrationEngine(config)
        assert engine._generator is not None

    @pytest.mark.unit
    def test_init_without_seed(self):
        """Test CalibrationEngine initialization without seed has no generator."""
        config = CalibrationConfig(seed=None)
        engine = CalibrationEngine(config)
        assert engine._generator is None


# =============================================================================
# CalibrationEngine Tests - Default Prompts
# =============================================================================


class TestCalibrationEngineDefaultPrompts:
    """Test CalibrationEngine default prompts functionality."""

    @pytest.mark.unit
    def test_get_default_prompts_returns_list(self):
        """Test get_default_prompts returns a list of strings."""
        config = CalibrationConfig()
        engine = CalibrationEngine(config)
        prompts = engine.get_default_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)

    @pytest.mark.unit
    def test_get_default_prompts_returns_copy(self):
        """Test get_default_prompts returns a copy, not the original."""
        config = CalibrationConfig()
        engine = CalibrationEngine(config)
        
        prompts1 = engine.get_default_prompts()
        prompts2 = engine.get_default_prompts()
        
        # Modify one copy
        prompts1.append("new prompt")
        
        # Other copy should be unaffected
        assert len(prompts2) < len(prompts1)

    @pytest.mark.unit
    def test_default_prompts_cover_diverse_domains(self):
        """Test default prompts cover various domains."""
        config = CalibrationConfig()
        engine = CalibrationEngine(config)
        prompts = engine.get_default_prompts()
        
        # Check for diversity in prompts
        all_prompts_text = " ".join(prompts).lower()
        
        # Should contain various domains
        assert "photo" in all_prompts_text or "portrait" in all_prompts_text
        assert "landscape" in all_prompts_text or "mountain" in all_prompts_text
        assert "art" in all_prompts_text or "painting" in all_prompts_text


# =============================================================================
# CalibrationEngine Tests - Random Prompt Generation
# =============================================================================


class TestCalibrationEngineRandomPrompts:
    """Test CalibrationEngine random prompt generation."""

    @pytest.mark.unit
    def test_generate_random_prompts_returns_correct_count(self):
        """Test generate_random_prompts returns requested number of prompts."""
        config = CalibrationConfig()
        engine = CalibrationEngine(config)
        
        prompts = engine.generate_random_prompts(10)
        assert len(prompts) == 10

    @pytest.mark.unit
    def test_generate_random_prompts_returns_strings(self):
        """Test generate_random_prompts returns list of strings."""
        config = CalibrationConfig()
        engine = CalibrationEngine(config)
        
        prompts = engine.generate_random_prompts(5)
        assert all(isinstance(p, str) for p in prompts)
        assert all(len(p) > 0 for p in prompts)

    @pytest.mark.unit
    def test_generate_random_prompts_zero(self):
        """Test generate_random_prompts with zero returns empty list."""
        config = CalibrationConfig()
        engine = CalibrationEngine(config)
        
        prompts = engine.generate_random_prompts(0)
        assert prompts == []

    @pytest.mark.unit
    def test_generate_random_prompts_large_count(self):
        """Test generate_random_prompts with large count."""
        config = CalibrationConfig()
        engine = CalibrationEngine(config)
        
        prompts = engine.generate_random_prompts(100)
        assert len(prompts) == 100


# =============================================================================
# Helper Functions for Mocking
# =============================================================================


def create_mock_tokenizer(model_max_length: int = 77):
    """Create a mock tokenizer for testing."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = model_max_length
    
    def tokenize_fn(prompts, **kwargs):
        batch_size = len(prompts) if isinstance(prompts, list) else 1
        mock_output = MagicMock()
        mock_output.input_ids = torch.randint(0, 1000, (batch_size, model_max_length))
        return mock_output
    
    mock_tokenizer.side_effect = tokenize_fn
    mock_tokenizer.__call__ = tokenize_fn
    return mock_tokenizer


def create_mock_text_encoder(device: str = "cpu", hidden_size: int = 768):
    """Create a mock text encoder for testing."""
    mock_encoder = MagicMock(spec=nn.Module)
    
    # Create a parameter to determine device
    mock_param = torch.nn.Parameter(torch.zeros(1, device=device))
    mock_encoder.parameters.return_value = iter([mock_param])
    
    def forward_fn(input_ids):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(
            batch_size, seq_len, hidden_size, 
            device=device, dtype=torch.float16
        )
        return mock_output
    
    mock_encoder.side_effect = forward_fn
    mock_encoder.__call__ = forward_fn
    return mock_encoder


# =============================================================================
# CalibrationEngine Tests - create_dataset() with Mocks
# =============================================================================


class TestCalibrationEngineCreateDataset:
    """Test CalibrationEngine.create_dataset() with mocked components."""

    @pytest.mark.unit
    def test_create_dataset_returns_iterator(self):
        """Test create_dataset returns an iterator."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        assert hasattr(dataset, '__iter__')
        assert hasattr(dataset, '__next__')

    @pytest.mark.unit
    def test_create_dataset_yields_correct_batch_structure(self):
        """Test create_dataset yields batches with correct keys."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch = next(dataset)
        
        # Verify batch structure (Requirement 2.3)
        assert "sample" in batch
        assert "timestep" in batch
        assert "encoder_hidden_states" in batch

    @pytest.mark.unit
    def test_create_dataset_sample_shape(self):
        """Test create_dataset yields samples with correct shape."""
        config = CalibrationConfig(
            num_samples=100,
            batch_size=1,
            image_size=(512, 512),
            latent_channels=4,
            latent_scale_factor=8,
        )
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch = next(dataset)
        
        # Expected shape: [batch_size, channels, height/8, width/8]
        expected_shape = (1, 4, 64, 64)
        assert batch["sample"].shape == expected_shape

    @pytest.mark.unit
    def test_create_dataset_timestep_shape(self):
        """Test create_dataset yields timesteps with correct shape."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch = next(dataset)
        
        # Timestep should be [batch_size]
        assert batch["timestep"].shape == (1,)
        assert batch["timestep"].dtype == torch.long

    @pytest.mark.unit
    def test_create_dataset_timestep_range(self):
        """Test create_dataset yields timesteps in valid range [0, 1000)."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        for batch in dataset:
            assert (batch["timestep"] >= 0).all()
            assert (batch["timestep"] < 1000).all()

    @pytest.mark.unit
    def test_create_dataset_encoder_hidden_states_shape(self):
        """Test create_dataset yields encoder_hidden_states with correct shape."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        hidden_size = 768
        seq_len = 77
        mock_tokenizer = create_mock_tokenizer(model_max_length=seq_len)
        mock_encoder = create_mock_text_encoder(hidden_size=hidden_size)
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch = next(dataset)
        
        # Expected shape: [batch_size, seq_len, hidden_size]
        assert batch["encoder_hidden_states"].shape[0] == 1
        assert batch["encoder_hidden_states"].shape[2] == hidden_size

    @pytest.mark.unit
    def test_create_dataset_sample_dtype(self):
        """Test create_dataset yields samples with FP16 dtype."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch = next(dataset)
        
        assert batch["sample"].dtype == torch.float16


# =============================================================================
# CalibrationEngine Tests - Streaming Iteration (Requirement 2.4)
# =============================================================================


class TestCalibrationEngineStreamingIteration:
    """Test CalibrationEngine streaming iteration to minimize memory footprint."""

    @pytest.mark.unit
    def test_streaming_yields_correct_number_of_batches(self):
        """Test streaming iteration yields correct number of batches."""
        num_samples = 100
        batch_size = 1
        config = CalibrationConfig(num_samples=num_samples, batch_size=batch_size)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * num_samples
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        batch_count = sum(1 for _ in dataset)
        expected_batches = (num_samples + batch_size - 1) // batch_size
        assert batch_count == expected_batches

    @pytest.mark.unit
    def test_streaming_with_larger_batch_size(self):
        """Test streaming with batch_size > 1."""
        num_samples = 100
        batch_size = 4
        config = CalibrationConfig(num_samples=num_samples, batch_size=batch_size)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * num_samples
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        batch_count = sum(1 for _ in dataset)
        expected_batches = (num_samples + batch_size - 1) // batch_size
        assert batch_count == expected_batches

    @pytest.mark.unit
    def test_streaming_last_batch_size(self):
        """Test last batch has correct size when samples not divisible by batch_size."""
        num_samples = 103  # Not divisible by 4
        batch_size = 4
        config = CalibrationConfig(num_samples=num_samples, batch_size=batch_size)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * num_samples
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        batches = list(dataset)
        
        # Last batch should have 103 % 4 = 3 samples
        last_batch = batches[-1]
        assert last_batch["sample"].shape[0] == 3

    @pytest.mark.unit
    def test_streaming_total_samples_generated(self):
        """Test total samples generated equals num_samples (Requirement 2.1)."""
        num_samples = 128
        batch_size = 8
        config = CalibrationConfig(num_samples=num_samples, batch_size=batch_size)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * num_samples
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        total_samples = sum(batch["sample"].shape[0] for batch in dataset)
        assert total_samples == num_samples

    @pytest.mark.unit
    def test_streaming_minimum_samples_requirement(self):
        """Test at least 100 samples are generated (Requirement 2.1)."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        total_samples = sum(batch["sample"].shape[0] for batch in dataset)
        assert total_samples >= 100


# =============================================================================
# CalibrationEngine Tests - Prompt Cycling
# =============================================================================


class TestCalibrationEnginePromptCycling:
    """Test CalibrationEngine prompt cycling when fewer prompts than samples."""

    @pytest.mark.unit
    def test_prompts_cycled_when_fewer_than_samples(self):
        """Test prompts are cycled when fewer prompts than num_samples."""
        num_samples = 100
        config = CalibrationConfig(num_samples=num_samples, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        # Only 10 prompts for 100 samples
        prompts = [f"prompt {i}" for i in range(10)]
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        # Should still generate 100 batches
        batch_count = sum(1 for _ in dataset)
        assert batch_count == num_samples

    @pytest.mark.unit
    def test_prompts_truncated_when_more_than_samples(self):
        """Test prompts are truncated when more prompts than num_samples."""
        num_samples = 100
        config = CalibrationConfig(num_samples=num_samples, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        # 200 prompts for 100 samples
        prompts = [f"prompt {i}" for i in range(200)]
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        # Should only generate 100 batches
        batch_count = sum(1 for _ in dataset)
        assert batch_count == num_samples


# =============================================================================
# CalibrationEngine Tests - collect_activations() with Mocked UNet
# =============================================================================


def create_mock_unet():
    """Create a mock UNet for testing collect_activations."""
    mock_unet = MagicMock(spec=nn.Module)
    
    # Create mock modules for named_modules
    mock_conv = MagicMock(spec=nn.Conv2d)
    mock_conv.children.return_value = iter([])  # Leaf module
    
    mock_linear = MagicMock(spec=nn.Linear)
    mock_linear.children.return_value = iter([])  # Leaf module
    
    def named_modules_fn():
        return iter([
            ("conv1", mock_conv),
            ("linear1", mock_linear),
        ])
    
    mock_unet.named_modules = named_modules_fn
    
    # Mock forward pass
    def forward_fn(sample, timestep, encoder_hidden_states=None):
        return torch.randn_like(sample)
    
    mock_unet.side_effect = forward_fn
    mock_unet.__call__ = forward_fn
    
    return mock_unet


class TestCalibrationEngineCollectActivations:
    """Test CalibrationEngine.collect_activations() with mocked UNet."""

    @pytest.mark.unit
    def test_collect_activations_sets_eval_mode(self):
        """Test collect_activations sets UNet to eval mode (Requirement 2.5)."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_unet = create_mock_unet()
        
        # Create simple calibration data
        def mock_calibration_data():
            for _ in range(2):
                yield {
                    "sample": torch.randn(1, 4, 64, 64),
                    "timestep": torch.tensor([500]),
                    "encoder_hidden_states": torch.randn(1, 77, 768),
                }
        
        engine.collect_activations(mock_unet, mock_calibration_data())
        
        mock_unet.eval.assert_called_once()

    @pytest.mark.unit
    def test_collect_activations_returns_dict(self):
        """Test collect_activations returns a dictionary."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_unet = create_mock_unet()
        
        def mock_calibration_data():
            for _ in range(2):
                yield {
                    "sample": torch.randn(1, 4, 64, 64),
                    "timestep": torch.tensor([500]),
                    "encoder_hidden_states": torch.randn(1, 77, 768),
                }
        
        result = engine.collect_activations(mock_unet, mock_calibration_data())
        
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_collect_activations_processes_all_batches(self):
        """Test collect_activations processes all calibration batches."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_unet = MagicMock(spec=nn.Module)
        mock_unet.named_modules.return_value = iter([])
        
        call_count = 0
        def forward_fn(sample, timestep, encoder_hidden_states=None):
            nonlocal call_count
            call_count += 1
            return torch.randn_like(sample)
        
        mock_unet.side_effect = forward_fn
        mock_unet.__call__ = forward_fn
        
        num_batches = 5
        def mock_calibration_data():
            for _ in range(num_batches):
                yield {
                    "sample": torch.randn(1, 4, 64, 64),
                    "timestep": torch.tensor([500]),
                    "encoder_hidden_states": torch.randn(1, 77, 768),
                }
        
        engine.collect_activations(mock_unet, mock_calibration_data())
        
        assert call_count == num_batches


# =============================================================================
# CalibrationEngine Tests - GPU-Dependent (marked with @pytest.mark.gpu)
# =============================================================================


class TestCalibrationEngineGPU:
    """GPU-dependent tests for CalibrationEngine. Run on Google Colab with T4."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_create_dataset_on_cuda(self):
        """Test create_dataset generates tensors on CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        # Create mock encoder on CUDA
        mock_encoder = MagicMock(spec=nn.Module)
        mock_param = torch.nn.Parameter(torch.zeros(1, device="cuda"))
        mock_encoder.parameters.return_value = iter([mock_param])
        
        def forward_fn(input_ids):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch_size, seq_len, 768, 
                device="cuda", dtype=torch.float16
            )
            return mock_output
        
        mock_encoder.__call__ = forward_fn
        mock_tokenizer = create_mock_tokenizer()
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch = next(dataset)
        
        assert batch["sample"].device.type == "cuda"
        assert batch["timestep"].device.type == "cuda"
        assert batch["encoder_hidden_states"].device.type == "cuda"

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_create_dataset_with_seed_reproducibility(self):
        """Test create_dataset with seed produces reproducible results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = CalibrationConfig(num_samples=100, batch_size=1, seed=42)
        
        # Create mock encoder on CUDA
        mock_encoder = MagicMock(spec=nn.Module)
        mock_param = torch.nn.Parameter(torch.zeros(1, device="cuda"))
        mock_encoder.parameters.return_value = iter([mock_param])
        
        def forward_fn(input_ids):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch_size, seq_len, 768, 
                device="cuda", dtype=torch.float16
            )
            return mock_output
        
        mock_encoder.__call__ = forward_fn
        mock_tokenizer = create_mock_tokenizer()
        prompts = ["test prompt"] * 100
        
        # First run
        engine1 = CalibrationEngine(config)
        dataset1 = engine1.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch1 = next(dataset1)
        
        # Second run with same seed
        config2 = CalibrationConfig(num_samples=100, batch_size=1, seed=42)
        engine2 = CalibrationEngine(config2)
        
        # Reset mock to return fresh parameters
        mock_encoder.parameters.return_value = iter([mock_param])
        
        dataset2 = engine2.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch2 = next(dataset2)
        
        # Latents should be identical with same seed
        assert torch.allclose(batch1["sample"], batch2["sample"])


# =============================================================================
# CalibrationEngine Tests - Text Encoder Integration (Requirement 2.2)
# =============================================================================


class TestCalibrationEngineTextEncoderIntegration:
    """Test CalibrationEngine uses pipeline's text encoder and tokenizer."""

    @pytest.mark.unit
    def test_uses_provided_tokenizer(self):
        """Test create_dataset uses the provided tokenizer (Requirement 2.2)."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 77
        
        def tokenize_fn(prompts, **kwargs):
            mock_output = MagicMock()
            batch_size = len(prompts) if isinstance(prompts, list) else 1
            mock_output.input_ids = torch.randint(0, 1000, (batch_size, 77))
            return mock_output
        
        mock_tokenizer.__call__ = tokenize_fn
        mock_encoder = create_mock_text_encoder()
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        _ = next(dataset)
        
        # Tokenizer should have been called
        mock_tokenizer.assert_called()

    @pytest.mark.unit
    def test_uses_provided_text_encoder(self):
        """Test create_dataset uses the provided text encoder (Requirement 2.2)."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = MagicMock(spec=nn.Module)
        
        mock_param = torch.nn.Parameter(torch.zeros(1))
        mock_encoder.parameters.return_value = iter([mock_param])
        
        call_count = 0
        def forward_fn(input_ids):
            nonlocal call_count
            call_count += 1
            batch_size = input_ids.shape[0]
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch_size, 77, 768, dtype=torch.float16
            )
            return mock_output
        
        mock_encoder.__call__ = forward_fn
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        _ = next(dataset)
        
        # Text encoder should have been called
        assert call_count > 0

    @pytest.mark.unit
    def test_text_encoder_set_to_eval_mode(self):
        """Test text encoder is set to eval mode during dataset creation."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = MagicMock(spec=nn.Module)
        
        mock_param = torch.nn.Parameter(torch.zeros(1))
        mock_encoder.parameters.return_value = iter([mock_param])
        
        def forward_fn(input_ids):
            batch_size = input_ids.shape[0]
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch_size, 77, 768, dtype=torch.float16
            )
            return mock_output
        
        mock_encoder.__call__ = forward_fn
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        _ = next(dataset)
        
        mock_encoder.eval.assert_called()


# =============================================================================
# CalibrationEngine Tests - Encoder Output Handling
# =============================================================================


class TestCalibrationEngineEncoderOutputHandling:
    """Test CalibrationEngine handles different text encoder output formats."""

    @pytest.mark.unit
    def test_handles_last_hidden_state_attribute(self):
        """Test handling encoder output with last_hidden_state attribute."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = MagicMock(spec=nn.Module)
        
        mock_param = torch.nn.Parameter(torch.zeros(1))
        mock_encoder.parameters.return_value = iter([mock_param])
        
        def forward_fn(input_ids):
            batch_size = input_ids.shape[0]
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(
                batch_size, 77, 768, dtype=torch.float16
            )
            return mock_output
        
        mock_encoder.__call__ = forward_fn
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch = next(dataset)
        
        assert "encoder_hidden_states" in batch
        assert batch["encoder_hidden_states"].shape[2] == 768

    @pytest.mark.unit
    def test_handles_tuple_output(self):
        """Test handling encoder output as tuple."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = MagicMock(spec=nn.Module)
        
        mock_param = torch.nn.Parameter(torch.zeros(1))
        mock_encoder.parameters.return_value = iter([mock_param])
        
        def forward_fn(input_ids):
            batch_size = input_ids.shape[0]
            # Return tuple instead of object with attribute
            hidden_states = torch.randn(batch_size, 77, 768, dtype=torch.float16)
            return (hidden_states, None)  # Tuple format
        
        mock_encoder.__call__ = forward_fn
        prompts = ["test prompt"] * 100
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        batch = next(dataset)
        
        assert "encoder_hidden_states" in batch
        assert batch["encoder_hidden_states"].shape[2] == 768


# =============================================================================
# Module Constants Tests
# =============================================================================


class TestModuleConstants:
    """Test module-level constants."""

    @pytest.mark.unit
    def test_default_calibration_prompts_exists(self):
        """Test DEFAULT_CALIBRATION_PROMPTS is defined."""
        assert DEFAULT_CALIBRATION_PROMPTS is not None
        assert isinstance(DEFAULT_CALIBRATION_PROMPTS, list)

    @pytest.mark.unit
    def test_default_calibration_prompts_not_empty(self):
        """Test DEFAULT_CALIBRATION_PROMPTS is not empty."""
        assert len(DEFAULT_CALIBRATION_PROMPTS) > 0

    @pytest.mark.unit
    def test_default_calibration_prompts_are_strings(self):
        """Test all DEFAULT_CALIBRATION_PROMPTS are strings."""
        assert all(isinstance(p, str) for p in DEFAULT_CALIBRATION_PROMPTS)

    @pytest.mark.unit
    def test_default_calibration_prompts_not_empty_strings(self):
        """Test no DEFAULT_CALIBRATION_PROMPTS are empty strings."""
        assert all(len(p) > 0 for p in DEFAULT_CALIBRATION_PROMPTS)


# =============================================================================
# Integration-style Tests with Mocks
# =============================================================================


class TestCalibrationEngineIntegration:
    """Integration-style tests for CalibrationEngine with comprehensive mocking."""

    @pytest.mark.unit
    def test_full_calibration_workflow(self):
        """Test complete workflow: create dataset, iterate, collect activations."""
        config = CalibrationConfig(num_samples=100, batch_size=4)
        engine = CalibrationEngine(config)
        
        # Setup mocks
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        prompts = engine.get_default_prompts()
        
        # Create dataset
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        # Verify we can iterate through all batches
        batches = list(dataset)
        
        # Verify batch count
        expected_batches = (config.num_samples + config.batch_size - 1) // config.batch_size
        assert len(batches) == expected_batches
        
        # Verify total samples
        total_samples = sum(b["sample"].shape[0] for b in batches)
        assert total_samples == config.num_samples

    @pytest.mark.unit
    def test_calibration_with_default_prompts(self):
        """Test calibration using default prompts."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        
        # Use default prompts
        prompts = engine.get_default_prompts()
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        # Should work without errors
        batch_count = sum(1 for _ in dataset)
        assert batch_count == config.num_samples

    @pytest.mark.unit
    def test_calibration_with_random_prompts(self):
        """Test calibration using randomly generated prompts."""
        config = CalibrationConfig(num_samples=100, batch_size=1)
        engine = CalibrationEngine(config)
        
        mock_tokenizer = create_mock_tokenizer()
        mock_encoder = create_mock_text_encoder()
        
        # Use random prompts
        prompts = engine.generate_random_prompts(50)
        
        dataset = engine.create_dataset(prompts, mock_encoder, mock_tokenizer)
        
        # Should work without errors (prompts will be cycled)
        batch_count = sum(1 for _ in dataset)
        assert batch_count == config.num_samples
