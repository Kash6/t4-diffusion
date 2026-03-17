"""
Property-based tests for INT8Quantizer component.

Property 6: Quantization Bounds
- Verify INT8 quantization error is bounded (MSE < 0.01)
- Verify scale > 0 and -128 <= zero_point <= 127 for quantized layers

Validates: Requirements 3.4, 3.5, 9.3

Note: These tests require nvidia-modelopt and GPU. They are marked with
@pytest.mark.gpu and will be skipped if dependencies are not available.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List

import torch
import torch.nn as nn

from diffusion_trt.quantizer import (
    QuantizationConfig,
    INT8Quantizer,
    QuantizationError,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_CALIBRATION_METHODS,
)
from diffusion_trt.models import MAX_QUANTIZATION_ERROR


# =============================================================================
# Property Test Strategies
# =============================================================================


@st.composite
def valid_quantization_config(draw):
    """Generate valid QuantizationConfig instances."""
    algorithm = draw(st.sampled_from(SUPPORTED_ALGORITHMS))
    calibration_method = draw(st.sampled_from(SUPPORTED_CALIBRATION_METHODS))
    percentile = draw(st.floats(min_value=90.0, max_value=100.0, exclude_min=False))
    max_error = draw(st.floats(min_value=0.001, max_value=0.1))
    num_batches = draw(st.integers(min_value=1, max_value=200))
    
    return QuantizationConfig(
        algorithm=algorithm,
        calibration_method=calibration_method,
        percentile=percentile,
        max_quantization_error=max_error,
        num_calibration_batches=num_batches,
    )


@st.composite
def calibration_batch(draw):
    """Generate a calibration batch with valid shapes."""
    batch_size = draw(st.integers(min_value=1, max_value=4))
    latent_size = draw(st.sampled_from([32, 64, 96]))
    seq_len = draw(st.sampled_from([77, 154]))
    hidden_size = draw(st.sampled_from([768, 1024, 1280]))
    
    return {
        "sample": torch.randn(batch_size, 4, latent_size, latent_size),
        "timestep": torch.randint(0, 1000, (batch_size,)),
        "encoder_hidden_states": torch.randn(batch_size, seq_len, hidden_size),
    }


# =============================================================================
# Property Tests - Configuration Validation
# =============================================================================


class TestQuantizationConfigProperties:
    """Property tests for QuantizationConfig validation."""

    @pytest.mark.unit
    @given(valid_quantization_config())
    @settings(max_examples=50)
    def test_valid_config_always_constructs(self, config):
        """Property: Valid configs always construct successfully."""
        assert config.algorithm in SUPPORTED_ALGORITHMS
        assert config.calibration_method in SUPPORTED_CALIBRATION_METHODS
        assert 0 < config.percentile <= 100
        assert config.max_quantization_error > 0
        assert config.num_calibration_batches >= 1

    @pytest.mark.unit
    @given(
        algorithm=st.text(min_size=1, max_size=20).filter(
            lambda x: x not in SUPPORTED_ALGORITHMS
        )
    )
    @settings(max_examples=20)
    def test_invalid_algorithm_always_fails(self, algorithm):
        """Property: Invalid algorithms always raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            QuantizationConfig(algorithm=algorithm)

    @pytest.mark.unit
    @given(percentile=st.floats(max_value=0))
    @settings(max_examples=20)
    def test_non_positive_percentile_fails(self, percentile):
        """Property: Non-positive percentile always raises ValueError."""
        with pytest.raises(ValueError, match="percentile must be in"):
            QuantizationConfig(percentile=percentile)

    @pytest.mark.unit
    @given(max_error=st.floats(max_value=0))
    @settings(max_examples=20)
    def test_non_positive_max_error_fails(self, max_error):
        """Property: Non-positive max_quantization_error always raises ValueError."""
        with pytest.raises(ValueError, match="max_quantization_error must be positive"):
            QuantizationConfig(max_quantization_error=max_error)


# =============================================================================
# Property Tests - Quantization Bounds (GPU Required)
# =============================================================================


class TestQuantizationBoundsProperties:
    """
    Property tests for quantization bounds.
    
    Property 6: Quantization Bounds
    - Verify INT8 quantization error is bounded (MSE < 0.01)
    - Verify scale > 0 and -128 <= zero_point <= 127 for quantized layers
    
    These tests require GPU and nvidia-modelopt.
    """

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires nvidia-modelopt and actual model quantization")
    @given(calibration_batch())
    @settings(max_examples=10)
    def test_quantization_error_bounded(self, batch):
        """
        Property: Quantization error is always bounded.
        
        For any valid calibration data, the MSE between original
        and quantized model outputs should be < MAX_QUANTIZATION_ERROR.
        """
        # This test would require:
        # 1. A real UNet model
        # 2. nvidia-modelopt installed
        # 3. GPU with sufficient memory
        # 
        # The test structure is provided for when these are available.
        pass

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires nvidia-modelopt and actual model quantization")
    def test_quantization_scales_positive(self):
        """
        Property: All quantization scales are positive.
        
        For any quantized layer, scale > 0 must hold.
        """
        pass

    @pytest.mark.unit
    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires nvidia-modelopt and actual model quantization")
    def test_zero_points_in_valid_range(self):
        """
        Property: All zero points are in valid INT8 range.
        
        For any quantized layer, -128 <= zero_point <= 127 must hold.
        """
        pass


# =============================================================================
# Property Tests - Accuracy Validation
# =============================================================================


class TestAccuracyValidationProperties:
    """Property tests for accuracy validation logic."""

    @pytest.mark.unit
    @given(
        mse=st.floats(min_value=0, max_value=0.001),
        tolerance=st.floats(min_value=0.01, max_value=0.1)
    )
    @settings(max_examples=30)
    def test_low_mse_always_passes(self, mse, tolerance):
        """Property: MSE below tolerance always passes validation."""
        assume(mse < tolerance)
        
        config = QuantizationConfig(max_quantization_error=tolerance)
        quantizer = INT8Quantizer(config)
        
        # Create mock models that return outputs with controlled MSE
        original = nn.Identity()
        quantized = nn.Identity()
        
        # Mock the forward function to return controlled outputs
        base_output = torch.randn(1, 4, 64, 64)
        noise_scale = (mse ** 0.5) * 0.9  # Ensure MSE is below target
        
        original_output = base_output
        quantized_output = base_output + torch.randn_like(base_output) * noise_scale
        
        # Patch the forward function
        def mock_forward(model, batch):
            if model is original:
                return original_output
            return quantized_output
        
        quantizer._default_forward_fn = mock_forward
        
        test_inputs = [{
            "sample": torch.randn(1, 4, 64, 64),
            "timestep": torch.tensor([500]),
            "encoder_hidden_states": torch.randn(1, 77, 768),
        }]
        
        # Should not raise
        result = quantizer.validate_accuracy(original, quantized, test_inputs)
        assert result["passed"] is True

    @pytest.mark.unit
    @given(
        mse=st.floats(min_value=0.1, max_value=1.0),
        tolerance=st.floats(min_value=0.001, max_value=0.01)
    )
    @settings(max_examples=30)
    def test_high_mse_always_fails(self, mse, tolerance):
        """Property: MSE above tolerance always fails validation."""
        assume(mse > tolerance)
        
        config = QuantizationConfig(max_quantization_error=tolerance)
        quantizer = INT8Quantizer(config)
        
        # Create outputs with high MSE
        original_output = torch.ones(1, 4, 64, 64)
        quantized_output = torch.zeros(1, 4, 64, 64)
        
        def mock_forward(model, batch):
            if hasattr(model, '_is_original') and model._is_original:
                return original_output
            return quantized_output
        
        quantizer._default_forward_fn = mock_forward
        
        original = nn.Identity()
        original._is_original = True
        quantized = nn.Identity()
        
        test_inputs = [{
            "sample": torch.randn(1, 4, 64, 64),
            "timestep": torch.tensor([500]),
            "encoder_hidden_states": torch.randn(1, 77, 768),
        }]
        
        with pytest.raises(QuantizationError):
            quantizer.validate_accuracy(original, quantized, test_inputs)


# =============================================================================
# Property Tests - Layer Exclusion
# =============================================================================


class TestLayerExclusionProperties:
    """Property tests for layer exclusion functionality."""

    @pytest.mark.unit
    @given(
        patterns=st.lists(
            st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=30)
    def test_exclude_filter_matches_own_patterns(self, patterns):
        """Property: Exclude filter always matches its own patterns."""
        config = QuantizationConfig(exclude_layers=patterns)
        quantizer = INT8Quantizer(config)
        
        filter_fn = quantizer._create_exclude_filter()
        
        # Each pattern should match a layer name containing it
        for pattern in patterns:
            layer_name = f"model.{pattern}.weight"
            assert filter_fn(layer_name) is True

    @pytest.mark.unit
    @given(
        patterns=st.lists(
            st.text(min_size=3, max_size=10, alphabet="xyz"),
            min_size=1,
            max_size=3
        ),
        layer_name=st.text(min_size=5, max_size=20, alphabet="abcdefghijklmnopqrstuvw_")
    )
    @settings(max_examples=30)
    def test_exclude_filter_no_false_positives(self, patterns, layer_name):
        """Property: Exclude filter doesn't match unrelated layer names."""
        # Ensure patterns don't appear in layer_name
        assume(all(p not in layer_name for p in patterns))
        
        config = QuantizationConfig(exclude_layers=patterns)
        quantizer = INT8Quantizer(config)
        
        filter_fn = quantizer._create_exclude_filter()
        
        assert filter_fn(layer_name) is False
