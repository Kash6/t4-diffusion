"""
Unit tests for ModelLoader component in diffusion_trt.model_loader.

Tests cover:
- ModelConfig validation (supported/unsupported models, device validation)
- ModelLoader.load() with mocked DiffusionPipeline
- ModelLoader.extract_unet() with mocked pipeline
- ModelLoader.get_vram_usage() and get_vram_info()
- OutOfMemoryError exception
- Retry logic for failed downloads (mocked)

Validates: Requirements 1.3, 1.4, 1.5
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import time

import torch

from diffusion_trt.model_loader import (
    ModelConfig,
    ModelLoader,
    OutOfMemoryError,
    SUPPORTED_MODELS,
    MAX_MODEL_WEIGHTS_VRAM_GB,
)
from diffusion_trt.models import T4_VRAM_LIMIT_GB


# =============================================================================
# ModelConfig Tests
# =============================================================================


class TestModelConfigValidConstruction:
    """Test valid construction of ModelConfig."""

    @pytest.mark.unit
    def test_valid_sdxl_turbo_config(self):
        """Test creating a valid ModelConfig for SDXL-Turbo."""
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        assert config.model_id == "stabilityai/sdxl-turbo"
        assert config.dtype == torch.float16
        assert config.variant == "fp16"
        assert config.device == "cuda"
        assert config.enable_attention_slicing is True
        assert config.enable_vae_tiling is True

    @pytest.mark.unit
    def test_valid_sd15_config(self):
        """Test creating a valid ModelConfig for SD 1.5."""
        config = ModelConfig(model_id="runwayml/stable-diffusion-v1-5")
        assert config.model_id == "runwayml/stable-diffusion-v1-5"

    @pytest.mark.unit
    def test_valid_config_with_custom_options(self):
        """Test ModelConfig with custom options."""
        config = ModelConfig(
            model_id="stabilityai/sdxl-turbo",
            dtype=torch.float32,
            variant=None,
            device="cpu",
            enable_attention_slicing=False,
            enable_vae_tiling=False,
        )
        assert config.dtype == torch.float32
        assert config.variant is None
        assert config.device == "cpu"
        assert config.enable_attention_slicing is False
        assert config.enable_vae_tiling is False


class TestModelConfigValidation:
    """Test validation errors for invalid ModelConfig inputs."""

    @pytest.mark.unit
    def test_invalid_unsupported_model(self):
        """Test that unsupported model_id raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model"):
            ModelConfig(model_id="unsupported/model-name")

    @pytest.mark.unit
    def test_invalid_model_with_supported_list_in_error(self):
        """Test that error message includes supported models list."""
        with pytest.raises(ValueError) as exc_info:
            ModelConfig(model_id="invalid/model")
        assert "stabilityai/sdxl-turbo" in str(exc_info.value)
        assert "runwayml/stable-diffusion-v1-5" in str(exc_info.value)

    @pytest.mark.unit
    def test_invalid_device(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            ModelConfig(
                model_id="stabilityai/sdxl-turbo",
                device="tpu",  # Invalid device
            )

    @pytest.mark.unit
    def test_invalid_device_empty_string(self):
        """Test that empty device string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            ModelConfig(
                model_id="stabilityai/sdxl-turbo",
                device="",
            )


# =============================================================================
# OutOfMemoryError Tests
# =============================================================================


class TestOutOfMemoryError:
    """Test OutOfMemoryError exception."""

    @pytest.mark.unit
    def test_error_message_format(self):
        """Test OutOfMemoryError message contains usage details."""
        error = OutOfMemoryError(
            current_usage_gb=16.5,
            limit_gb=15.6,
            operation="model loading"
        )
        assert "16.50GB" in error.message
        assert "15.60GB" in error.message
        assert "model loading" in error.message

    @pytest.mark.unit
    def test_error_attributes(self):
        """Test OutOfMemoryError stores attributes correctly."""
        error = OutOfMemoryError(
            current_usage_gb=17.0,
            limit_gb=15.6,
            operation="inference"
        )
        assert error.current_usage_gb == 17.0
        assert error.limit_gb == 15.6

    @pytest.mark.unit
    def test_error_default_limit(self):
        """Test OutOfMemoryError uses T4 limit by default."""
        error = OutOfMemoryError(current_usage_gb=16.0)
        assert error.limit_gb == T4_VRAM_LIMIT_GB

    @pytest.mark.unit
    def test_error_is_exception(self):
        """Test OutOfMemoryError can be raised and caught."""
        with pytest.raises(OutOfMemoryError) as exc_info:
            raise OutOfMemoryError(current_usage_gb=16.0, operation="test")
        assert exc_info.value.current_usage_gb == 16.0


# =============================================================================
# ModelLoader Tests - Mocked Loading
# =============================================================================


class TestModelLoaderLoad:
    """Test ModelLoader.load() with mocked DiffusionPipeline."""

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)  # 8GB
    @patch('torch.cuda.empty_cache')
    def test_load_applies_memory_optimizations(
        self, mock_empty_cache, mock_mem_alloc, mock_cuda_avail, mock_get_pipeline
    ):
        """Test that load() applies attention slicing and VAE tiling."""
        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        loader = ModelLoader()
        
        result = loader.load(config)
        
        # Verify memory optimizations were applied
        mock_pipeline.enable_attention_slicing.assert_called_once()
        mock_pipeline.enable_vae_tiling.assert_called_once()
        mock_pipeline.to.assert_called_once_with("cuda")

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)
    @patch('torch.cuda.empty_cache')
    def test_load_skips_optimizations_when_disabled(
        self, mock_empty_cache, mock_mem_alloc, mock_cuda_avail, mock_get_pipeline
    ):
        """Test that load() skips optimizations when disabled in config."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        config = ModelConfig(
            model_id="stabilityai/sdxl-turbo",
            enable_attention_slicing=False,
            enable_vae_tiling=False,
        )
        loader = ModelLoader()
        
        loader.load(config)
        
        mock_pipeline.enable_attention_slicing.assert_not_called()
        mock_pipeline.enable_vae_tiling.assert_not_called()

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)
    @patch('torch.cuda.empty_cache')
    def test_load_uses_fp16_dtype(
        self, mock_empty_cache, mock_mem_alloc, mock_cuda_avail, mock_get_pipeline
    ):
        """Test that load() uses FP16 dtype by default."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        loader = ModelLoader()
        
        loader.load(config)
        
        # Verify from_pretrained was called with correct dtype
        call_kwargs = mock_pipeline_cls.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.float16
        assert call_kwargs["variant"] == "fp16"

    @pytest.mark.unit
    def test_load_raises_for_unsupported_model(self):
        """Test that load() raises ValueError for unsupported models."""
        loader = ModelLoader()
        
        # Create config with a supported model first, then modify
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        config.model_id = "unsupported/model"  # Bypass __post_init__
        
        with pytest.raises(ValueError, match="Unsupported model"):
            loader.load(config)

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=17 * 1024**3)  # 17GB - exceeds limit
    @patch('torch.cuda.empty_cache')
    def test_load_raises_oom_when_vram_exceeded(
        self, mock_empty_cache, mock_mem_alloc, mock_cuda_avail, mock_get_pipeline
    ):
        """Test that load() raises OutOfMemoryError when VRAM exceeds limit."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        loader = ModelLoader()
        
        with pytest.raises(OutOfMemoryError) as exc_info:
            loader.load(config)
        
        assert exc_info.value.current_usage_gb > T4_VRAM_LIMIT_GB
        assert "model loading" in exc_info.value.message


# =============================================================================
# ModelLoader Tests - Retry Logic
# =============================================================================


class TestModelLoaderRetryLogic:
    """Test ModelLoader retry logic for failed downloads."""

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)
    @patch('torch.cuda.empty_cache')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_retry_on_download_failure(
        self, mock_sleep, mock_empty_cache, mock_mem_alloc, mock_cuda_avail, mock_get_pipeline
    ):
        """Test that load() retries on download failure."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls = MagicMock()
        # Fail twice, then succeed
        mock_pipeline_cls.from_pretrained.side_effect = [
            ConnectionError("Network error"),
            ConnectionError("Network error"),
            mock_pipeline,
        ]
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        loader = ModelLoader(max_retries=3)
        
        result = loader.load(config)
        
        assert mock_pipeline_cls.from_pretrained.call_count == 3
        assert result == mock_pipeline

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('time.sleep')
    def test_retry_uses_exponential_backoff(
        self, mock_sleep, mock_empty_cache, mock_cuda_avail, mock_get_pipeline
    ):
        """Test that retry uses exponential backoff timing."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.side_effect = [
            ConnectionError("Error 1"),
            ConnectionError("Error 2"),
            mock_pipeline,
        ]
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        loader = ModelLoader(max_retries=3)
        
        with patch('torch.cuda.memory_allocated', return_value=8 * 1024**3):
            loader.load(config)
        
        # Verify exponential backoff: 1s, 2s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)  # 2^0 = 1
        mock_sleep.assert_any_call(2)  # 2^1 = 2

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('time.sleep')
    def test_raises_after_max_retries(
        self, mock_sleep, mock_empty_cache, mock_cuda_avail, mock_get_pipeline
    ):
        """Test that load() raises RuntimeError after max retries."""
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.side_effect = ConnectionError("Persistent error")
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        loader = ModelLoader(max_retries=3)
        
        with pytest.raises(RuntimeError) as exc_info:
            loader.load(config)
        
        assert "Failed to load model" in str(exc_info.value)
        assert "3 attempts" in str(exc_info.value)
        assert mock_pipeline_cls.from_pretrained.call_count == 3


# =============================================================================
# ModelLoader Tests - Extract UNet
# =============================================================================


class TestModelLoaderExtractUnet:
    """Test ModelLoader.extract_unet() with mocked pipeline."""

    @pytest.mark.unit
    def test_extract_unet_returns_unet_module(self):
        """Test that extract_unet() returns the UNet from pipeline."""
        mock_unet = MagicMock(spec=torch.nn.Module)
        mock_pipeline = MagicMock()
        mock_pipeline.unet = mock_unet
        
        loader = ModelLoader()
        result = loader.extract_unet(mock_pipeline)
        
        assert result == mock_unet

    @pytest.mark.unit
    def test_extract_unet_sets_eval_mode(self):
        """Test that extract_unet() sets UNet to eval mode."""
        mock_unet = MagicMock(spec=torch.nn.Module)
        mock_pipeline = MagicMock()
        mock_pipeline.unet = mock_unet
        
        loader = ModelLoader()
        loader.extract_unet(mock_pipeline)
        
        mock_unet.eval.assert_called_once()

    @pytest.mark.unit
    def test_extract_unet_raises_for_missing_unet(self):
        """Test that extract_unet() raises AttributeError if no UNet."""
        mock_pipeline = MagicMock(spec=[])  # No unet attribute
        del mock_pipeline.unet  # Ensure unet doesn't exist
        
        loader = ModelLoader()
        
        with pytest.raises(AttributeError, match="does not have a 'unet' attribute"):
            loader.extract_unet(mock_pipeline)


# =============================================================================
# ModelLoader Tests - VRAM Monitoring
# =============================================================================


class TestModelLoaderVRAMMonitoring:
    """Test ModelLoader VRAM monitoring methods."""

    @pytest.mark.unit
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=10 * 1024**3)  # 10GB
    def test_get_vram_usage_returns_gb(self, mock_mem_alloc, mock_cuda_avail):
        """Test that get_vram_usage() returns usage in GB."""
        loader = ModelLoader()
        usage = loader.get_vram_usage()
        
        assert usage == pytest.approx(10.0, rel=0.01)

    @pytest.mark.unit
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_vram_usage_returns_zero_without_cuda(self, mock_cuda_avail):
        """Test that get_vram_usage() returns 0.0 without CUDA."""
        loader = ModelLoader()
        usage = loader.get_vram_usage()
        
        assert usage == 0.0

    @pytest.mark.unit
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_reserved', return_value=12 * 1024**3)  # 12GB
    def test_get_vram_reserved_returns_gb(self, mock_mem_reserved, mock_cuda_avail):
        """Test that get_vram_reserved() returns reserved memory in GB."""
        loader = ModelLoader()
        reserved = loader.get_vram_reserved()
        
        assert reserved == pytest.approx(12.0, rel=0.01)

    @pytest.mark.unit
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_vram_reserved_returns_zero_without_cuda(self, mock_cuda_avail):
        """Test that get_vram_reserved() returns 0.0 without CUDA."""
        loader = ModelLoader()
        reserved = loader.get_vram_reserved()
        
        assert reserved == 0.0

    @pytest.mark.unit
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)  # 8GB
    @patch('torch.cuda.memory_reserved', return_value=10 * 1024**3)  # 10GB
    @patch('torch.cuda.get_device_properties')
    def test_get_vram_info_returns_dict(
        self, mock_props, mock_reserved, mock_allocated, mock_cuda_avail
    ):
        """Test that get_vram_info() returns complete VRAM info dict."""
        mock_props.return_value.total_memory = 16 * 1024**3  # 16GB total
        
        loader = ModelLoader()
        info = loader.get_vram_info()
        
        assert "allocated_gb" in info
        assert "reserved_gb" in info
        assert "total_gb" in info
        assert "free_gb" in info
        assert info["allocated_gb"] == pytest.approx(8.0, rel=0.01)
        assert info["reserved_gb"] == pytest.approx(10.0, rel=0.01)
        assert info["total_gb"] == pytest.approx(16.0, rel=0.01)
        assert info["free_gb"] == pytest.approx(8.0, rel=0.01)  # 16 - 8

    @pytest.mark.unit
    @patch('torch.cuda.is_available', return_value=False)
    def test_get_vram_info_returns_zeros_without_cuda(self, mock_cuda_avail):
        """Test that get_vram_info() returns zeros without CUDA."""
        loader = ModelLoader()
        info = loader.get_vram_info()
        
        assert info["allocated_gb"] == 0.0
        assert info["reserved_gb"] == 0.0
        assert info["total_gb"] == 0.0
        assert info["free_gb"] == 0.0


# =============================================================================
# ModelLoader Tests - Memory Management
# =============================================================================


class TestModelLoaderMemoryManagement:
    """Test ModelLoader memory management methods."""

    @pytest.mark.unit
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_clear_memory_empties_cache(
        self, mock_gc, mock_empty_cache, mock_cuda_avail
    ):
        """Test that clear_memory() empties CUDA cache."""
        loader = ModelLoader()
        loader.clear_memory()
        
        mock_empty_cache.assert_called_once()
        mock_gc.assert_called_once()

    @pytest.mark.unit
    @patch('torch.cuda.is_available', return_value=False)
    @patch('gc.collect')
    def test_clear_memory_works_without_cuda(self, mock_gc, mock_cuda_avail):
        """Test that clear_memory() works without CUDA."""
        loader = ModelLoader()
        loader.clear_memory()  # Should not raise
        
        mock_gc.assert_called_once()

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_clear_memory_clears_loaded_pipeline(
        self, mock_gc, mock_empty_cache, mock_mem_alloc, mock_cuda_avail, mock_get_pipeline
    ):
        """Test that clear_memory() clears the loaded pipeline reference."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        loader = ModelLoader()
        loader.load(config)
        
        assert loader._loaded_pipeline is not None
        
        loader.clear_memory()
        
        assert loader._loaded_pipeline is None


# =============================================================================
# ModelLoader Tests - GPU-Dependent (marked with @pytest.mark.gpu)
# =============================================================================


class TestModelLoaderGPU:
    """GPU-dependent tests for ModelLoader. Run on Google Colab with T4."""

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_vram_usage_real_gpu(self):
        """Test get_vram_usage() with real GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        loader = ModelLoader()
        usage = loader.get_vram_usage()
        
        # Should return a non-negative value
        assert usage >= 0.0
        # Should be less than T4 total memory
        assert usage < 20.0  # Reasonable upper bound

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_get_vram_info_real_gpu(self):
        """Test get_vram_info() with real GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        loader = ModelLoader()
        info = loader.get_vram_info()
        
        # All values should be non-negative
        assert info["allocated_gb"] >= 0.0
        assert info["reserved_gb"] >= 0.0
        assert info["total_gb"] > 0.0
        assert info["free_gb"] >= 0.0
        
        # Total should be reasonable for a GPU
        assert info["total_gb"] > 1.0
        assert info["total_gb"] < 100.0

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_vram_usage_increases_with_tensor_allocation(self):
        """Test that VRAM usage increases when allocating tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        loader = ModelLoader()
        
        # Get baseline usage
        baseline = loader.get_vram_usage()
        
        # Allocate a large tensor (100MB)
        tensor = torch.randn(25 * 1024 * 1024, device="cuda")  # ~100MB
        
        # Check usage increased
        after_alloc = loader.get_vram_usage()
        assert after_alloc > baseline
        
        # Clean up
        del tensor
        torch.cuda.empty_cache()

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_clear_memory_reduces_vram(self):
        """Test that clear_memory() reduces VRAM usage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        loader = ModelLoader()
        
        # Allocate some tensors
        tensors = [torch.randn(1024 * 1024, device="cuda") for _ in range(10)]
        
        usage_before = loader.get_vram_usage()
        
        # Delete tensors and clear memory
        del tensors
        loader.clear_memory()
        
        usage_after = loader.get_vram_usage()
        
        # Usage should decrease or stay same (not increase)
        assert usage_after <= usage_before


# =============================================================================
# Integration-style Tests with Mocks
# =============================================================================


class TestModelLoaderIntegration:
    """Integration-style tests for ModelLoader with comprehensive mocking."""

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)
    @patch('torch.cuda.empty_cache')
    def test_full_load_and_extract_workflow(
        self, mock_empty_cache, mock_mem_alloc, mock_cuda_avail, mock_get_pipeline
    ):
        """Test complete workflow: load pipeline, extract UNet."""
        # Setup mock pipeline with UNet
        mock_unet = MagicMock(spec=torch.nn.Module)
        mock_pipeline = MagicMock()
        mock_pipeline.unet = mock_unet
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        # Execute workflow
        config = ModelConfig(model_id="stabilityai/sdxl-turbo")
        loader = ModelLoader()
        
        pipeline = loader.load(config)
        unet = loader.extract_unet(pipeline)
        
        # Verify
        assert pipeline == mock_pipeline
        assert unet == mock_unet
        mock_unet.eval.assert_called_once()

    @pytest.mark.unit
    @patch('diffusion_trt.model_loader._get_diffusion_pipeline')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)
    @patch('torch.cuda.empty_cache')
    def test_load_both_supported_models(
        self, mock_empty_cache, mock_mem_alloc, mock_cuda_avail, mock_get_pipeline
    ):
        """Test loading both supported models."""
        mock_pipeline = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline
        mock_get_pipeline.return_value = mock_pipeline_cls
        
        loader = ModelLoader()
        
        # Load SDXL-Turbo
        config1 = ModelConfig(model_id="stabilityai/sdxl-turbo")
        result1 = loader.load(config1)
        assert result1 is not None
        
        # Load SD 1.5
        config2 = ModelConfig(model_id="runwayml/stable-diffusion-v1-5")
        result2 = loader.load(config2)
        assert result2 is not None
        
        # Verify both models were loaded
        assert mock_pipeline_cls.from_pretrained.call_count == 2


# =============================================================================
# Constants and Module-Level Tests
# =============================================================================


class TestModuleConstants:
    """Test module-level constants."""

    @pytest.mark.unit
    def test_supported_models_list(self):
        """Test SUPPORTED_MODELS contains expected models."""
        assert "stabilityai/sdxl-turbo" in SUPPORTED_MODELS
        assert "runwayml/stable-diffusion-v1-5" in SUPPORTED_MODELS
        assert len(SUPPORTED_MODELS) == 2

    @pytest.mark.unit
    def test_max_model_weights_vram(self):
        """Test MAX_MODEL_WEIGHTS_VRAM_GB is reasonable."""
        assert MAX_MODEL_WEIGHTS_VRAM_GB == 10.0
        assert MAX_MODEL_WEIGHTS_VRAM_GB < T4_VRAM_LIMIT_GB
