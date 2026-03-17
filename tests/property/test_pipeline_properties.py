"""
Property-based tests for OptimizedPipeline component.

Property 5: Deterministic Output
- Verify same seed produces identical output for same prompt

Validates: Requirements 6.3, 12.3

Note: These tests use mocks to avoid downloading actual models while
verifying the seed/generator mechanism works correctly.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Tuple, List
from unittest.mock import Mock, MagicMock, patch

import torch

from diffusion_trt.pipeline import PipelineConfig, OptimizedPipeline


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


# =============================================================================
# Property Test Strategies
# =============================================================================


@st.composite
def valid_seed(draw):
    """Generate valid random seeds for deterministic output testing."""
    # Use a reasonable range for seeds
    return draw(st.integers(min_value=0, max_value=2**31 - 1))


@st.composite
def valid_prompt(draw):
    """Generate valid text prompts for testing."""
    # Generate prompts with various characteristics
    base_prompts = [
        "A photo of a cat",
        "A beautiful sunset over mountains",
        "A futuristic city at night",
        "An abstract painting with vibrant colors",
        "A serene lake surrounded by trees",
        "A portrait of a person smiling",
        "A detailed illustration of a dragon",
        "A minimalist design with geometric shapes",
    ]
    
    prompt = draw(st.sampled_from(base_prompts))
    
    # Optionally add style modifiers
    add_modifier = draw(st.booleans())
    if add_modifier:
        modifiers = [
            ", high quality",
            ", 4k resolution",
            ", detailed",
            ", professional",
            ", artistic",
        ]
        modifier = draw(st.sampled_from(modifiers))
        prompt = prompt + modifier
    
    return prompt


@st.composite
def inference_params(draw):
    """Generate valid inference parameters."""
    num_inference_steps = draw(st.integers(min_value=1, max_value=20))
    guidance_scale = draw(st.floats(min_value=0.0, max_value=15.0))
    
    return {
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
    }


# =============================================================================
# Mock Helpers
# =============================================================================


def create_mock_diffusers_pipeline(seed: int, prompt: str, **kwargs):
    """
    Create a mock diffusers pipeline that produces deterministic outputs.
    
    The mock simulates the behavior of a real diffusers pipeline by using
    the seed to generate reproducible tensor outputs.
    """
    mock_pipeline = MagicMock()
    
    def mock_call(prompt, generator=None, **call_kwargs):
        """Mock __call__ that produces deterministic output based on generator."""
        # Create deterministic output based on generator state
        if generator is not None:
            # Use the generator to create reproducible output
            # This simulates how a real pipeline would use the generator
            output_tensor = torch.randn(
                1, 3, 64, 64,
                generator=generator,
                device=generator.device if hasattr(generator, 'device') else 'cpu',
            )
        else:
            # Non-deterministic output without generator
            output_tensor = torch.randn(1, 3, 64, 64)
        
        # Create mock result with images attribute
        mock_result = MagicMock()
        mock_result.images = [output_tensor]
        return mock_result
    
    mock_pipeline.side_effect = mock_call
    return mock_pipeline


# =============================================================================
# Property Tests - Deterministic Output (Property 5)
# =============================================================================


class TestDeterministicOutputProperties:
    """
    Property tests for deterministic output.
    
    Property 5: Deterministic Output
    - Verify same seed produces identical output for same prompt
    
    Validates: Requirements 6.3, 12.3
    """

    @pytest.mark.gpu
    @given(
        seed=valid_seed(),
        prompt=valid_prompt(),
    )
    @settings(max_examples=50, deadline=None)
    def test_same_seed_produces_identical_output(
        self,
        seed: int,
        prompt: str,
    ):
        """
        **Validates: Requirements 6.3, 12.3**
        
        Property: Same seed produces identical output for the same prompt.
        
        For any valid seed and prompt, running inference twice with the same
        seed should produce pixel-wise identical outputs.
        """
        # Create pipeline config
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            seed=None,  # We'll pass seed explicitly to __call__
        )
        pipeline = OptimizedPipeline(config)
        
        # Setup mock diffusers pipeline
        mock_diffusers = MagicMock()
        
        # Track generator states for verification
        generator_states = []
        
        def mock_call(prompt, generator=None, **kwargs):
            """Mock that captures generator state and produces deterministic output."""
            if generator is not None:
                # Capture the initial state of the generator
                initial_state = generator.get_state().clone()
                generator_states.append(initial_state)
                
                # Generate output using the generator
                output = torch.randn(
                    1, 3, 64, 64,
                    generator=generator,
                    device=generator.device,
                )
            else:
                output = torch.randn(1, 3, 64, 64, device="cuda")
            
            mock_result = MagicMock()
            mock_result.images = [output]
            return mock_result
        
        mock_diffusers.side_effect = mock_call
        pipeline._pipeline = mock_diffusers
        
        # First inference with seed
        output1 = pipeline(prompt, seed=seed)
        
        # Second inference with same seed
        output2 = pipeline(prompt, seed=seed)
        
        # Verify both outputs are tensors
        assert len(output1) == 1
        assert len(output2) == 1
        
        # Verify generator states were captured
        assert len(generator_states) == 2
        
        # Verify generator states are identical (same seed produces same initial state)
        assert torch.equal(generator_states[0], generator_states[1]), (
            f"Generator states differ for seed {seed}. "
            f"Same seed should produce identical generator states."
        )

    @pytest.mark.gpu
    @given(
        seed=valid_seed(),
        prompt=valid_prompt(),
        params=inference_params(),
    )
    @settings(max_examples=30, deadline=None)
    def test_determinism_with_various_inference_params(
        self,
        seed: int,
        prompt: str,
        params: dict,
    ):
        """
        **Validates: Requirements 6.3, 12.3**
        
        Property: Determinism holds regardless of inference parameters.
        
        For any valid seed, prompt, and inference parameters (num_steps, guidance_scale),
        running inference twice with the same seed should produce identical outputs.
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Track outputs for comparison
        outputs = []
        
        def mock_call(prompt, generator=None, num_inference_steps=None, 
                      guidance_scale=None, **kwargs):
            """Mock that produces deterministic output based on generator."""
            if generator is not None:
                output = torch.randn(
                    1, 3, 64, 64,
                    generator=generator,
                    device=generator.device,
                )
            else:
                output = torch.randn(1, 3, 64, 64, device="cuda")
            
            outputs.append(output.clone())
            
            mock_result = MagicMock()
            mock_result.images = [output]
            return mock_result
        
        mock_diffusers = MagicMock()
        mock_diffusers.side_effect = mock_call
        pipeline._pipeline = mock_diffusers
        
        # First inference
        _ = pipeline(
            prompt,
            seed=seed,
            num_inference_steps=params['num_inference_steps'],
            guidance_scale=params['guidance_scale'],
        )
        
        # Second inference with same parameters
        _ = pipeline(
            prompt,
            seed=seed,
            num_inference_steps=params['num_inference_steps'],
            guidance_scale=params['guidance_scale'],
        )
        
        # Verify outputs are identical
        assert len(outputs) == 2
        assert torch.equal(outputs[0], outputs[1]), (
            f"Outputs differ for seed {seed} with params {params}. "
            f"Max diff: {(outputs[0] - outputs[1]).abs().max().item()}"
        )

    @pytest.mark.gpu
    @given(
        seed1=valid_seed(),
        seed2=valid_seed(),
        prompt=valid_prompt(),
    )
    @settings(max_examples=30, deadline=None)
    def test_different_seeds_produce_different_outputs(
        self,
        seed1: int,
        seed2: int,
        prompt: str,
    ):
        """
        **Validates: Requirements 6.3, 12.3**
        
        Property: Different seeds produce different outputs.
        
        For any two different seeds and the same prompt, the outputs should
        be different (with very high probability for random outputs).
        """
        # Skip if seeds are the same
        assume(seed1 != seed2)
        
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        outputs = []
        
        def mock_call(prompt, generator=None, **kwargs):
            """Mock that produces deterministic output based on generator."""
            if generator is not None:
                output = torch.randn(
                    1, 3, 64, 64,
                    generator=generator,
                    device=generator.device,
                )
            else:
                output = torch.randn(1, 3, 64, 64, device="cuda")
            
            outputs.append(output.clone())
            
            mock_result = MagicMock()
            mock_result.images = [output]
            return mock_result
        
        mock_diffusers = MagicMock()
        mock_diffusers.side_effect = mock_call
        pipeline._pipeline = mock_diffusers
        
        # First inference with seed1
        _ = pipeline(prompt, seed=seed1)
        
        # Second inference with seed2
        _ = pipeline(prompt, seed=seed2)
        
        # Verify outputs are different
        assert len(outputs) == 2
        assert not torch.equal(outputs[0], outputs[1]), (
            f"Outputs are identical for different seeds {seed1} and {seed2}. "
            f"Different seeds should produce different outputs."
        )

    @pytest.mark.gpu
    @given(
        seed=valid_seed(),
        prompts=st.lists(valid_prompt(), min_size=2, max_size=5, unique=True),
    )
    @settings(max_examples=20, deadline=None)
    def test_determinism_across_multiple_prompts(
        self,
        seed: int,
        prompts: List[str],
    ):
        """
        **Validates: Requirements 6.3, 12.3**
        
        Property: Determinism holds across multiple different prompts.
        
        For a given seed, running inference on multiple prompts twice should
        produce identical outputs for each prompt.
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Store outputs for each prompt
        first_run_outputs = {}
        second_run_outputs = {}
        current_run = [1]  # Use list to allow modification in nested function
        
        def mock_call(prompt, generator=None, **kwargs):
            """Mock that produces deterministic output based on generator and prompt."""
            if generator is not None:
                # Use prompt hash to vary output per prompt while maintaining determinism
                prompt_hash = hash(prompt) % 1000
                torch.manual_seed(generator.initial_seed() + prompt_hash)
                output = torch.randn(1, 3, 64, 64, device=generator.device)
            else:
                output = torch.randn(1, 3, 64, 64, device="cuda")
            
            if current_run[0] == 1:
                first_run_outputs[prompt] = output.clone()
            else:
                second_run_outputs[prompt] = output.clone()
            
            mock_result = MagicMock()
            mock_result.images = [output]
            return mock_result
        
        mock_diffusers = MagicMock()
        mock_diffusers.side_effect = mock_call
        pipeline._pipeline = mock_diffusers
        
        # First run through all prompts
        for prompt in prompts:
            _ = pipeline(prompt, seed=seed)
        
        # Second run through all prompts
        current_run[0] = 2
        for prompt in prompts:
            _ = pipeline(prompt, seed=seed)
        
        # Verify outputs match for each prompt
        for prompt in prompts:
            assert prompt in first_run_outputs, f"Missing first run output for '{prompt}'"
            assert prompt in second_run_outputs, f"Missing second run output for '{prompt}'"
            
            assert torch.equal(first_run_outputs[prompt], second_run_outputs[prompt]), (
                f"Outputs differ for prompt '{prompt}' with seed {seed}. "
                f"Max diff: {(first_run_outputs[prompt] - second_run_outputs[prompt]).abs().max().item()}"
            )

    @pytest.mark.gpu
    @given(seed=valid_seed())
    @settings(max_examples=30, deadline=None)
    def test_generator_setup_is_deterministic(self, seed: int):
        """
        **Validates: Requirements 6.3, 12.3**
        
        Property: Generator setup with same seed produces identical initial state.
        
        The _setup_generator method should produce identical generator states
        when called with the same seed.
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        
        # Create two pipelines and setup generators with same seed
        pipeline1 = OptimizedPipeline(config)
        pipeline2 = OptimizedPipeline(config)
        
        pipeline1._setup_generator(seed)
        pipeline2._setup_generator(seed)
        
        # Verify generators exist
        assert pipeline1._generator is not None
        assert pipeline2._generator is not None
        
        # Verify generator states are identical
        state1 = pipeline1._generator.get_state()
        state2 = pipeline2._generator.get_state()
        
        assert torch.equal(state1, state2), (
            f"Generator states differ for seed {seed}. "
            f"_setup_generator should produce identical states for same seed."
        )

    @pytest.mark.gpu
    @given(
        seed=valid_seed(),
        prompt=valid_prompt(),
    )
    @settings(max_examples=30, deadline=None)
    def test_config_seed_produces_deterministic_output(
        self,
        seed: int,
        prompt: str,
    ):
        """
        **Validates: Requirements 6.3, 12.3**
        
        Property: Seed provided in config produces deterministic output.
        
        When seed is provided in PipelineConfig, the pipeline should produce
        deterministic outputs without needing to pass seed to __call__.
        """
        # Create pipeline with seed in config
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            seed=seed,
        )
        pipeline = OptimizedPipeline(config)
        
        # Verify generator was set up during initialization
        assert pipeline._generator is not None
        
        # Capture generator state
        initial_state = pipeline._generator.get_state().clone()
        
        # Create another pipeline with same config
        config2 = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            seed=seed,
        )
        pipeline2 = OptimizedPipeline(config2)
        
        # Verify both pipelines have identical generator states
        assert torch.equal(initial_state, pipeline2._generator.get_state()), (
            f"Pipelines with same seed {seed} in config have different generator states."
        )

    @pytest.mark.gpu
    @given(
        seed=valid_seed(),
        prompt=valid_prompt(),
        num_runs=st.integers(min_value=3, max_value=5),
    )
    @settings(max_examples=20, deadline=None)
    def test_determinism_across_multiple_runs(
        self,
        seed: int,
        prompt: str,
        num_runs: int,
    ):
        """
        **Validates: Requirements 6.3, 12.3**
        
        Property: Determinism holds across multiple consecutive runs.
        
        Running inference multiple times with the same seed should always
        produce identical outputs, not just for two runs.
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        outputs = []
        
        def mock_call(prompt, generator=None, **kwargs):
            """Mock that produces deterministic output based on generator."""
            if generator is not None:
                output = torch.randn(
                    1, 3, 64, 64,
                    generator=generator,
                    device=generator.device,
                )
            else:
                output = torch.randn(1, 3, 64, 64, device="cuda")
            
            outputs.append(output.clone())
            
            mock_result = MagicMock()
            mock_result.images = [output]
            return mock_result
        
        mock_diffusers = MagicMock()
        mock_diffusers.side_effect = mock_call
        pipeline._pipeline = mock_diffusers
        
        # Run inference multiple times with same seed
        for _ in range(num_runs):
            _ = pipeline(prompt, seed=seed)
        
        # Verify all outputs are identical
        assert len(outputs) == num_runs
        
        reference_output = outputs[0]
        for i, output in enumerate(outputs[1:], start=2):
            assert torch.equal(reference_output, output), (
                f"Output {i} differs from output 1 for seed {seed}. "
                f"Max diff: {(reference_output - output).abs().max().item()}"
            )
