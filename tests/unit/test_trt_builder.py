"""
Unit tests for TensorRTBuilder component in diffusion_trt.trt_builder.

Tests cover:
- TRTConfig validation
- TensorRTBuilder initialization
- Compilation with mocked torch_tensorrt
- Engine serialization/loading
- Engine metadata

Validates: Requirements 4.5, 4.6
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json
import tempfile
import os

import torch
import torch.nn as nn

from diffusion_trt.trt_builder import (
    TRTConfig,
    TensorRTBuilder,
    SUPPORTED_PRECISIONS,
    T4_COMPUTE_CAPABILITY,
    DEFAULT_WORKSPACE_SIZE,
)


# =============================================================================
# TRTConfig Tests - Valid Construction
# =============================================================================


class TestTRTConfigValidConstruction:
    """Test valid construction of TRTConfig."""

    @pytest.mark.unit
    def test_valid_default_config(self):
        """Test creating a TRTConfig with default values."""
        config = TRTConfig()
        assert config.precision == "int8"
        assert config.workspace_size == DEFAULT_WORKSPACE_SIZE
        assert config.max_batch_size == 1
        assert config.optimization_level == 5
        assert config.use_cuda_graph is True
        assert config.dynamic_shapes is False
        assert config.target_device == T4_COMPUTE_CAPABILITY

    @pytest.mark.unit
    def test_valid_fp32_precision(self):
        """Test TRTConfig with fp32 precision."""
        config = TRTConfig(precision="fp32")
        assert config.precision == "fp32"

    @pytest.mark.unit
    def test_valid_fp16_precision(self):
        """Test TRTConfig with fp16 precision."""
        config = TRTConfig(precision="fp16")
        assert config.precision == "fp16"

    @pytest.mark.unit
    def test_valid_int8_precision(self):
        """Test TRTConfig with int8 precision."""
        config = TRTConfig(precision="int8")
        assert config.precision == "int8"

    @pytest.mark.unit
    def test_valid_custom_workspace(self):
        """Test TRTConfig with custom workspace size."""
        config = TRTConfig(workspace_size=8 * 1024 * 1024 * 1024)  # 8GB
        assert config.workspace_size == 8 * 1024 * 1024 * 1024

    @pytest.mark.unit
    def test_valid_optimization_levels(self):
        """Test TRTConfig with various optimization levels."""
        for level in range(6):
            config = TRTConfig(optimization_level=level)
            assert config.optimization_level == level

    @pytest.mark.unit
    def test_valid_batch_size(self):
        """Test TRTConfig with custom batch size."""
        config = TRTConfig(max_batch_size=4)
        assert config.max_batch_size == 4

    @pytest.mark.unit
    def test_valid_cuda_graph_disabled(self):
        """Test TRTConfig with CUDA graphs disabled."""
        config = TRTConfig(use_cuda_graph=False)
        assert config.use_cuda_graph is False

    @pytest.mark.unit
    def test_valid_dynamic_shapes_enabled(self):
        """Test TRTConfig with dynamic shapes enabled."""
        config = TRTConfig(dynamic_shapes=True)
        assert config.dynamic_shapes is True


# =============================================================================
# TRTConfig Tests - Validation Errors
# =============================================================================


class TestTRTConfigValidation:
    """Test validation errors for invalid TRTConfig inputs."""

    @pytest.mark.unit
    def test_invalid_precision(self):
        """Test that unsupported precision raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported precision"):
            TRTConfig(precision="bf16")

    @pytest.mark.unit
    def test_invalid_workspace_zero(self):
        """Test that workspace_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="workspace_size must be positive"):
            TRTConfig(workspace_size=0)

    @pytest.mark.unit
    def test_invalid_workspace_negative(self):
        """Test that negative workspace_size raises ValueError."""
        with pytest.raises(ValueError, match="workspace_size must be positive"):
            TRTConfig(workspace_size=-1024)

    @pytest.mark.unit
    def test_invalid_batch_size_zero(self):
        """Test that max_batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_batch_size must be at least 1"):
            TRTConfig(max_batch_size=0)

    @pytest.mark.unit
    def test_invalid_batch_size_negative(self):
        """Test that negative max_batch_size raises ValueError."""
        with pytest.raises(ValueError, match="max_batch_size must be at least 1"):
            TRTConfig(max_batch_size=-1)

    @pytest.mark.unit
    def test_invalid_optimization_level_negative(self):
        """Test that negative optimization_level raises ValueError."""
        with pytest.raises(ValueError, match="optimization_level must be in"):
            TRTConfig(optimization_level=-1)

    @pytest.mark.unit
    def test_invalid_optimization_level_too_high(self):
        """Test that optimization_level > 5 raises ValueError."""
        with pytest.raises(ValueError, match="optimization_level must be in"):
            TRTConfig(optimization_level=6)


# =============================================================================
# TensorRTBuilder Tests - Initialization
# =============================================================================


class TestTensorRTBuilderInit:
    """Test TensorRTBuilder initialization."""

    @pytest.mark.unit
    def test_init_with_default_config(self):
        """Test TensorRTBuilder initialization with default config."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        assert builder.config == config
        assert builder._compiled_model is None

    @pytest.mark.unit
    def test_init_with_custom_config(self):
        """Test TensorRTBuilder initialization with custom config."""
        config = TRTConfig(
            precision="fp16",
            optimization_level=3,
            use_cuda_graph=False
        )
        builder = TensorRTBuilder(config)
        assert builder.config.precision == "fp16"
        assert builder.config.optimization_level == 3


# =============================================================================
# TensorRTBuilder Tests - Compilation with Mocks
# =============================================================================


class TestTensorRTBuilderCompile:
    """Test TensorRTBuilder.compile_torchtrt() with mocked torch_tensorrt."""

    @pytest.mark.unit
    @patch('torch.compile')
    def test_compile_calls_torch_compile(self, mock_compile):
        """Test that compile_torchtrt calls torch.compile."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        mock_model = MagicMock(spec=nn.Module)
        mock_compiled = MagicMock()
        mock_compile.return_value = mock_compiled
        
        sample_inputs = [torch.randn(1, 4, 64, 64)]
        
        with patch.dict('sys.modules', {'torch_tensorrt': MagicMock()}):
            result = builder.compile_torchtrt(mock_model, sample_inputs)
        
        mock_compile.assert_called_once()
        mock_model.eval.assert_called()

    @pytest.mark.unit
    def test_compile_sets_eval_mode(self):
        """Test that compile_torchtrt sets model to eval mode."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        mock_model = MagicMock(spec=nn.Module)
        sample_inputs = [torch.randn(1, 4, 64, 64)]
        
        with patch.dict('sys.modules', {'torch_tensorrt': MagicMock()}):
            with patch('torch.compile', return_value=MagicMock()):
                try:
                    builder.compile_torchtrt(mock_model, sample_inputs)
                except:
                    pass
        
        mock_model.eval.assert_called()

    @pytest.mark.unit
    def test_compile_raises_without_torch_tensorrt(self):
        """Test that compile_torchtrt raises ImportError without torch_tensorrt."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        mock_model = MagicMock(spec=nn.Module)
        sample_inputs = [torch.randn(1, 4, 64, 64)]
        
        # Ensure torch_tensorrt is not importable
        with patch.dict('sys.modules', {'torch_tensorrt': None}):
            with pytest.raises(ImportError, match="torch-tensorrt is required"):
                builder.compile_torchtrt(mock_model, sample_inputs)


# =============================================================================
# TensorRTBuilder Tests - Build Settings
# =============================================================================


class TestTensorRTBuilderSettings:
    """Test TensorRTBuilder compile settings generation."""

    @pytest.mark.unit
    def test_build_settings_fp32(self):
        """Test compile settings for fp32 precision."""
        config = TRTConfig(precision="fp32")
        builder = TensorRTBuilder(config)
        
        sample_inputs = [torch.randn(1, 4, 64, 64)]
        settings = builder._build_compile_settings(sample_inputs)
        
        assert torch.float32 in settings["enabled_precisions"]
        assert torch.float16 not in settings["enabled_precisions"]

    @pytest.mark.unit
    def test_build_settings_fp16(self):
        """Test compile settings for fp16 precision."""
        config = TRTConfig(precision="fp16")
        builder = TensorRTBuilder(config)
        
        sample_inputs = [torch.randn(1, 4, 64, 64)]
        settings = builder._build_compile_settings(sample_inputs)
        
        assert torch.float16 in settings["enabled_precisions"]
        assert torch.float32 in settings["enabled_precisions"]

    @pytest.mark.unit
    def test_build_settings_int8(self):
        """Test compile settings for int8 precision."""
        config = TRTConfig(precision="int8")
        builder = TensorRTBuilder(config)
        
        sample_inputs = [torch.randn(1, 4, 64, 64)]
        settings = builder._build_compile_settings(sample_inputs)
        
        assert torch.int8 in settings["enabled_precisions"]
        assert torch.float16 in settings["enabled_precisions"]

    @pytest.mark.unit
    def test_build_settings_workspace(self):
        """Test compile settings include workspace size."""
        config = TRTConfig(workspace_size=2 * 1024 * 1024 * 1024)
        builder = TensorRTBuilder(config)
        
        sample_inputs = [torch.randn(1, 4, 64, 64)]
        settings = builder._build_compile_settings(sample_inputs)
        
        assert settings["workspace_size"] == 2 * 1024 * 1024 * 1024

    @pytest.mark.unit
    def test_build_settings_dynamic_shapes(self):
        """Test compile settings with dynamic shapes enabled."""
        config = TRTConfig(dynamic_shapes=True)
        builder = TensorRTBuilder(config)
        
        sample_inputs = [torch.randn(1, 4, 64, 64)]
        settings = builder._build_compile_settings(sample_inputs)
        
        assert settings.get("dynamic_batch") is True


# =============================================================================
# TensorRTBuilder Tests - Engine Info
# =============================================================================


class TestTensorRTBuilderEngineInfo:
    """Test TensorRTBuilder engine info methods."""

    @pytest.mark.unit
    def test_get_engine_info_file_not_found(self):
        """Test get_engine_info raises FileNotFoundError for missing file."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        with pytest.raises(FileNotFoundError):
            builder.get_engine_info("/nonexistent/path/engine.trt")

    @pytest.mark.unit
    def test_get_engine_info_with_metadata(self):
        """Test get_engine_info loads metadata from JSON file."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy engine file
            engine_path = Path(tmpdir) / "test.trt"
            engine_path.write_bytes(b"dummy engine data")
            
            # Create metadata file
            metadata = {
                "precision": "int8",
                "optimization_level": 5,
                "tensorrt_version": "8.6.0"
            }
            metadata_path = engine_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            info = builder.get_engine_info(str(engine_path))
            
            assert info["precision"] == "int8"
            assert info["optimization_level"] == 5
            assert info["tensorrt_version"] == "8.6.0"
            assert "file_size_mb" in info

    @pytest.mark.unit
    def test_get_engine_info_without_metadata(self):
        """Test get_engine_info works without metadata file."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy engine file without metadata
            engine_path = Path(tmpdir) / "test.trt"
            engine_path.write_bytes(b"dummy engine data")
            
            info = builder.get_engine_info(str(engine_path))
            
            assert "file_size_mb" in info
            assert "engine_path" in info


# =============================================================================
# TensorRTBuilder Tests - Engine Compatibility
# =============================================================================


class TestTensorRTBuilderCompatibility:
    """Test TensorRTBuilder engine compatibility validation."""

    @pytest.mark.unit
    def test_validate_compatibility_same_version(self):
        """Test compatibility check passes for same TensorRT version."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine_path = Path(tmpdir) / "test.trt"
            engine_path.write_bytes(b"dummy")
            
            metadata = {"tensorrt_version": "8.6.0"}
            with open(engine_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f)
            
            is_compatible, msg = builder.validate_engine_compatibility(
                str(engine_path),
                current_trt_version="8.6.1"
            )
            
            assert is_compatible is True

    @pytest.mark.unit
    def test_validate_compatibility_different_major_version(self):
        """Test compatibility check fails for different major version."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine_path = Path(tmpdir) / "test.trt"
            engine_path.write_bytes(b"dummy")
            
            metadata = {"tensorrt_version": "7.2.0"}
            with open(engine_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f)
            
            is_compatible, msg = builder.validate_engine_compatibility(
                str(engine_path),
                current_trt_version="8.6.0"
            )
            
            assert is_compatible is False
            assert "mismatch" in msg.lower()

    @pytest.mark.unit
    def test_validate_compatibility_no_version_info(self):
        """Test compatibility check passes when no version info available."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine_path = Path(tmpdir) / "test.trt"
            engine_path.write_bytes(b"dummy")
            
            # No metadata file
            is_compatible, msg = builder.validate_engine_compatibility(
                str(engine_path),
                current_trt_version="8.6.0"
            )
            
            assert is_compatible is True


# =============================================================================
# TensorRTBuilder Tests - Load Engine
# =============================================================================


class TestTensorRTBuilderLoadEngine:
    """Test TensorRTBuilder.load_engine()."""

    @pytest.mark.unit
    def test_load_engine_file_not_found(self):
        """Test load_engine raises FileNotFoundError for missing file."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        with patch.dict('sys.modules', {'torch_tensorrt': MagicMock()}):
            with pytest.raises(FileNotFoundError):
                builder.load_engine("/nonexistent/path/engine.trt")

    @pytest.mark.unit
    def test_load_engine_raises_without_torch_tensorrt(self):
        """Test load_engine raises ImportError without torch_tensorrt."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine_path = Path(tmpdir) / "test.trt"
            engine_path.write_bytes(b"dummy")
            
            with patch.dict('sys.modules', {'torch_tensorrt': None}):
                with pytest.raises(ImportError, match="torch-tensorrt is required"):
                    builder.load_engine(str(engine_path))


# =============================================================================
# TensorRTBuilder Tests - Get Compiled Model
# =============================================================================


class TestTensorRTBuilderGetModel:
    """Test TensorRTBuilder.get_compiled_model()."""

    @pytest.mark.unit
    def test_get_compiled_model_none_initially(self):
        """Test get_compiled_model returns None before compilation."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        assert builder.get_compiled_model() is None

    @pytest.mark.unit
    def test_get_compiled_model_after_compile(self):
        """Test get_compiled_model returns model after compilation."""
        config = TRTConfig()
        builder = TensorRTBuilder(config)
        
        mock_compiled = MagicMock()
        builder._compiled_model = mock_compiled
        
        assert builder.get_compiled_model() == mock_compiled


# =============================================================================
# Module Constants Tests
# =============================================================================


class TestModuleConstants:
    """Test module-level constants."""

    @pytest.mark.unit
    def test_supported_precisions(self):
        """Test SUPPORTED_PRECISIONS contains expected values."""
        assert "fp32" in SUPPORTED_PRECISIONS
        assert "fp16" in SUPPORTED_PRECISIONS
        assert "int8" in SUPPORTED_PRECISIONS

    @pytest.mark.unit
    def test_t4_compute_capability(self):
        """Test T4_COMPUTE_CAPABILITY is correct."""
        assert T4_COMPUTE_CAPABILITY == "sm_75"

    @pytest.mark.unit
    def test_default_workspace_size(self):
        """Test DEFAULT_WORKSPACE_SIZE is 4GB."""
        assert DEFAULT_WORKSPACE_SIZE == 4 * 1024 * 1024 * 1024
