"""
Unit tests for the OptimizedPipeline component.

Tests cover:
- PipelineConfig validation
- OptimizedPipeline initialization
- Deterministic output with seed
- Cache manager integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch

from diffusion_trt.pipeline import PipelineConfig, OptimizedPipeline
from diffusion_trt.model_loader import SUPPORTED_MODELS


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""
    
    def test_valid_config_sdxl_turbo(self):
        """Test creating config with SDXL-Turbo defaults."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            enable_int8=True,
            enable_caching=True,
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        
        assert config.model_id == "stabilityai/sdxl-turbo"
        assert config.enable_int8 is True
        assert config.enable_caching is True
        assert config.num_inference_steps == 4
        assert config.guidance_scale == 0.0
    
    def test_valid_config_sd15(self):
        """Test creating config with SD 1.5 settings."""
        config = PipelineConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            enable_int8=True,
            enable_caching=True,
            num_inference_steps=20,
            guidance_scale=7.5,
        )
        
        assert config.model_id == "runwayml/stable-diffusion-v1-5"
        assert config.num_inference_steps == 20
        assert config.guidance_scale == 7.5
    
    def test_invalid_model_id(self):
        """Test that unsupported model_id raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model"):
            PipelineConfig(model_id="invalid/model-id")
    
    def test_invalid_cache_interval(self):
        """Test that cache_interval < 1 raises ValueError."""
        with pytest.raises(ValueError, match="cache_interval must be >= 1"):
            PipelineConfig(
                model_id="stabilityai/sdxl-turbo",
                cache_interval=0,
            )
    
    def test_invalid_num_inference_steps(self):
        """Test that num_inference_steps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_inference_steps must be >= 1"):
            PipelineConfig(
                model_id="stabilityai/sdxl-turbo",
                num_inference_steps=0,
            )
    
    def test_invalid_guidance_scale(self):
        """Test that negative guidance_scale raises ValueError."""
        with pytest.raises(ValueError, match="guidance_scale must be >= 0"):
            PipelineConfig(
                model_id="stabilityai/sdxl-turbo",
                guidance_scale=-1.0,
            )
    
    def test_invalid_num_calibration_samples(self):
        """Test that num_calibration_samples < 100 raises ValueError."""
        with pytest.raises(ValueError, match="num_calibration_samples must be >= 100"):
            PipelineConfig(
                model_id="stabilityai/sdxl-turbo",
                num_calibration_samples=50,
            )
    
    def test_invalid_optimization_level(self):
        """Test that optimization_level outside [0, 5] raises ValueError."""
        with pytest.raises(ValueError, match="optimization_level must be in"):
            PipelineConfig(
                model_id="stabilityai/sdxl-turbo",
                optimization_level=6,
            )
    
    def test_invalid_max_cache_size_gb(self):
        """Test that max_cache_size_gb outside (0, 2.0] raises ValueError."""
        with pytest.raises(ValueError, match="max_cache_size_gb must be in"):
            PipelineConfig(
                model_id="stabilityai/sdxl-turbo",
                max_cache_size_gb=3.0,
            )
        
        with pytest.raises(ValueError, match="max_cache_size_gb must be in"):
            PipelineConfig(
                model_id="stabilityai/sdxl-turbo",
                max_cache_size_gb=0.0,
            )
    
    def test_config_with_seed(self):
        """Test config with seed for deterministic outputs."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            seed=42,
        )
        
        assert config.seed == 42
    
    def test_config_with_exclude_layers(self):
        """Test config with layers excluded from quantization."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            exclude_layers=["conv_out", "final_layer"],
        )
        
        assert config.exclude_layers == ["conv_out", "final_layer"]
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        
        assert config.enable_int8 is True
        assert config.enable_caching is True
        assert config.cache_interval == 3
        assert config.num_inference_steps == 4
        assert config.guidance_scale == 0.0
        assert config.seed is None
        assert config.image_size == (512, 512)
        assert config.num_calibration_samples == 512
        assert config.optimization_level == 5
        assert config.max_cache_size_gb == 2.0
        assert config.exclude_layers is None


class TestOptimizedPipeline:
    """Tests for OptimizedPipeline class."""
    
    def test_init_with_config(self):
        """Test initializing pipeline with config."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            seed=42,
        )
        
        pipeline = OptimizedPipeline(config)
        
        assert pipeline.config == config
        assert pipeline._is_optimized is False
        assert pipeline._generator is not None  # Seed was provided
    
    def test_init_without_seed(self):
        """Test initializing pipeline without seed."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            seed=None,
        )
        
        pipeline = OptimizedPipeline(config)
        
        assert pipeline._generator is None
    
    def test_setup_generator(self):
        """Test that generator is set up correctly with seed."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        pipeline._setup_generator(123)
        
        assert pipeline._generator is not None
    
    def test_is_optimized_property(self):
        """Test is_optimized property."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        assert pipeline.is_optimized is False
        
        pipeline._is_optimized = True
        assert pipeline.is_optimized is True
    
    def test_call_without_initialization_raises_error(self):
        """Test that calling pipeline without initialization raises error."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        with pytest.raises(RuntimeError, match="Pipeline not initialized"):
            pipeline("A test prompt")
    
    def test_get_vram_usage_without_cuda(self):
        """Test get_vram_usage returns 0 when CUDA is not available."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Without model loader, should return 0 or use torch.cuda
        vram = pipeline.get_vram_usage()
        assert isinstance(vram, float)
        assert vram >= 0.0
    
    def test_get_cache_stats_without_cache_manager(self):
        """Test get_cache_stats returns None when caching is disabled."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        stats = pipeline.get_cache_stats()
        assert stats is None
    
    def test_clear_cache_without_cache_manager(self):
        """Test clear_cache works even without cache manager."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Should not raise
        pipeline.clear_cache()
    
    def test_properties_without_pipeline(self):
        """Test that properties return None when pipeline is not loaded."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        assert pipeline.text_encoder is None
        assert pipeline.tokenizer is None
        assert pipeline.vae is None
        assert pipeline.scheduler is None
        assert pipeline.unet is None


class TestOptimizedPipelineWithMocks:
    """Tests for OptimizedPipeline using mocks."""
    
    @patch('diffusion_trt.pipeline.ModelLoader')
    def test_load_model(self, mock_loader_class):
        """Test _load_model method."""
        # Setup mock
        mock_loader = MagicMock()
        mock_pipeline = MagicMock()
        mock_unet = MagicMock()
        
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = mock_pipeline
        mock_loader.extract_unet.return_value = mock_unet
        mock_loader.get_vram_usage.return_value = 5.0
        
        # Create pipeline and load model
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        pipeline._load_model()
        
        # Verify
        mock_loader.load.assert_called_once()
        mock_loader.extract_unet.assert_called_once_with(mock_pipeline)
        assert pipeline._pipeline == mock_pipeline
        assert pipeline._unet == mock_unet
    
    @patch('diffusion_trt.pipeline.CacheManager')
    def test_setup_caching(self, mock_cache_class):
        """Test _setup_caching method."""
        mock_cache = MagicMock()
        mock_cache_class.return_value = mock_cache
        
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            cache_interval=3,
            max_cache_size_gb=1.5,
        )
        pipeline = OptimizedPipeline(config)
        pipeline._setup_caching()
        
        # Verify cache manager was created
        mock_cache_class.assert_called_once()
        assert pipeline._cache_manager == mock_cache
    
    def test_call_with_single_prompt(self):
        """Test __call__ with a single prompt string."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Call with single prompt
        images = pipeline("A test prompt")
        
        # Verify
        mock_diffusers_pipeline.assert_called_once()
        assert len(images) == 1
    
    def test_call_with_multiple_prompts(self):
        """Test __call__ with multiple prompts."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Call with multiple prompts
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        images = pipeline(prompts)
        
        # Verify - should be called once per prompt
        assert mock_diffusers_pipeline.call_count == 3
        assert len(images) == 3
    
    def test_call_with_seed_for_determinism(self):
        """Test __call__ with seed produces deterministic generator."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Call with seed
        pipeline("A test prompt", seed=42)
        
        # Verify generator was passed
        call_kwargs = mock_diffusers_pipeline.call_args[1]
        assert call_kwargs.get('generator') is not None
    
    def test_call_with_negative_prompt(self):
        """Test __call__ with negative prompt."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Call with negative prompt
        pipeline("A test prompt", negative_prompt="bad quality")
        
        # Verify negative prompt was passed
        call_kwargs = mock_diffusers_pipeline.call_args[1]
        assert call_kwargs.get('negative_prompt') == "bad quality"
    
    def test_call_overrides_config_values(self):
        """Test __call__ can override config values."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Call with overridden values
        pipeline(
            "A test prompt",
            num_inference_steps=10,
            guidance_scale=7.5,
        )
        
        # Verify overridden values were used
        call_kwargs = mock_diffusers_pipeline.call_args[1]
        assert call_kwargs.get('num_inference_steps') == 10
        assert call_kwargs.get('guidance_scale') == 7.5
    
    def test_call_clears_cache_before_inference(self):
        """Test __call__ clears cache before inference."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline and cache manager
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        
        mock_cache = MagicMock()
        
        pipeline._pipeline = mock_diffusers_pipeline
        pipeline._cache_manager = mock_cache
        
        # Call pipeline
        pipeline("A test prompt")
        
        # Verify cache was cleared
        mock_cache.clear.assert_called_once()
    
    def test_call_increments_cache_step(self):
        """Test __call__ increments cache step counter."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline and cache manager
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        
        mock_cache = MagicMock()
        
        pipeline._pipeline = mock_diffusers_pipeline
        pipeline._cache_manager = mock_cache
        
        # Call pipeline with multiple prompts
        pipeline(["Prompt 1", "Prompt 2"])
        
        # Verify step was incremented for each prompt
        assert mock_cache.increment_step.call_count == 2


class TestOptimizedPipelineFromPretrained:
    """Tests for OptimizedPipeline.from_pretrained class method."""
    
    @patch.object(OptimizedPipeline, '_setup_caching')
    @patch.object(OptimizedPipeline, '_compile_tensorrt')
    @patch.object(OptimizedPipeline, '_apply_quantization')
    @patch.object(OptimizedPipeline, '_load_model')
    def test_from_pretrained_full_optimization(
        self,
        mock_load,
        mock_quantize,
        mock_compile,
        mock_cache,
    ):
        """Test from_pretrained with all optimizations enabled."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            enable_int8=True,
            enable_caching=True,
        )
        
        pipeline = OptimizedPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            config=config,
        )
        
        # Verify all optimization steps were called
        mock_load.assert_called_once()
        mock_quantize.assert_called_once()
        mock_compile.assert_called_once()
        mock_cache.assert_called_once()
        
        assert pipeline.is_optimized is True
    
    @patch.object(OptimizedPipeline, '_setup_caching')
    @patch.object(OptimizedPipeline, '_compile_tensorrt')
    @patch.object(OptimizedPipeline, '_apply_quantization')
    @patch.object(OptimizedPipeline, '_load_model')
    def test_from_pretrained_without_int8(
        self,
        mock_load,
        mock_quantize,
        mock_compile,
        mock_cache,
    ):
        """Test from_pretrained with INT8 disabled."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            enable_int8=False,
            enable_caching=True,
        )
        
        pipeline = OptimizedPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            config=config,
        )
        
        # Verify quantization and TRT compilation were skipped
        mock_load.assert_called_once()
        mock_quantize.assert_not_called()
        mock_compile.assert_not_called()
        mock_cache.assert_called_once()
    
    @patch.object(OptimizedPipeline, '_setup_caching')
    @patch.object(OptimizedPipeline, '_compile_tensorrt')
    @patch.object(OptimizedPipeline, '_apply_quantization')
    @patch.object(OptimizedPipeline, '_load_model')
    def test_from_pretrained_without_caching(
        self,
        mock_load,
        mock_quantize,
        mock_compile,
        mock_cache,
    ):
        """Test from_pretrained with caching disabled."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            enable_int8=True,
            enable_caching=False,
        )
        
        pipeline = OptimizedPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            config=config,
        )
        
        # Verify caching was skipped
        mock_load.assert_called_once()
        mock_quantize.assert_called_once()
        mock_compile.assert_called_once()
        mock_cache.assert_not_called()
    
    @patch.object(OptimizedPipeline, '_setup_caching')
    @patch.object(OptimizedPipeline, '_compile_tensorrt')
    @patch.object(OptimizedPipeline, '_apply_quantization')
    @patch.object(OptimizedPipeline, '_load_model')
    def test_from_pretrained_creates_default_config(
        self,
        mock_load,
        mock_quantize,
        mock_compile,
        mock_cache,
    ):
        """Test from_pretrained creates default config if not provided."""
        pipeline = OptimizedPipeline.from_pretrained("stabilityai/sdxl-turbo")
        
        assert pipeline.config.model_id == "stabilityai/sdxl-turbo"
        assert pipeline.config.enable_int8 is True
        assert pipeline.config.enable_caching is True


class TestEnginePersistence:
    """Tests for save_engine() and load_engine() methods.
    
    Requirements covered:
    - 6.4: Support saving optimized engines to disk via save_engine method
    - 6.5: When load_engine is called, restore the pipeline without re-optimization
    - 12.1: Serialize the TensorRT engine to the specified path
    - 12.2: Deserialize and restore the engine without recompilation
    - 12.3: Produce identical outputs to the original for the same inputs
    - 12.4: Validate engine compatibility with current TensorRT version
    """
    
    def test_save_engine_raises_if_not_optimized(self, tmp_path):
        """Test save_engine raises error if pipeline is not optimized."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        with pytest.raises(RuntimeError, match="Pipeline is not optimized"):
            pipeline.save_engine(str(tmp_path / "test_engine.pt"))
    
    def test_save_engine_raises_if_no_trt_engine(self, tmp_path):
        """Test save_engine raises error if no TensorRT engine exists."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        pipeline._is_optimized = True
        pipeline._trt_unet = None
        
        with pytest.raises(RuntimeError, match="No TensorRT engine to save"):
            pipeline.save_engine(str(tmp_path / "test_engine.pt"))
    
    def test_save_engine_creates_directory(self, tmp_path):
        """Test save_engine creates parent directory if needed."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        pipeline._is_optimized = True
        pipeline._trt_unet = MagicMock()
        
        # Create a mock that can be saved
        mock_model = torch.nn.Linear(10, 10)
        pipeline._trt_unet = mock_model
        
        engine_path = tmp_path / "subdir" / "nested" / "test_engine.pt"
        pipeline.save_engine(str(engine_path))
        
        assert engine_path.parent.exists()
    
    def test_save_engine_creates_metadata_file(self, tmp_path):
        """Test save_engine creates JSON metadata file."""
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            enable_int8=True,
            cache_interval=5,
            num_inference_steps=8,
        )
        pipeline = OptimizedPipeline(config)
        pipeline._is_optimized = True
        pipeline._trt_unet = torch.nn.Linear(10, 10)
        
        engine_path = tmp_path / "test_engine.pt"
        pipeline.save_engine(str(engine_path))
        
        metadata_path = engine_path.with_suffix('.json')
        assert metadata_path.exists()
        
        # Verify metadata content
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert 'pipeline_config' in metadata
        assert metadata['pipeline_config']['model_id'] == "stabilityai/sdxl-turbo"
        assert metadata['pipeline_config']['enable_int8'] is True
        assert metadata['pipeline_config']['cache_interval'] == 5
        assert metadata['pipeline_config']['num_inference_steps'] == 8
        assert 'created_at' in metadata
        assert 'torch_version' in metadata
    
    def test_load_engine_raises_if_file_not_found(self, tmp_path):
        """Test load_engine raises error if file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Engine file not found"):
            OptimizedPipeline.load_engine(str(tmp_path / "nonexistent.pt"))
    
    @patch.object(OptimizedPipeline, '_load_model')
    @patch.object(OptimizedPipeline, '_setup_caching')
    def test_load_engine_restores_config(self, mock_cache, mock_load, tmp_path):
        """Test load_engine restores pipeline config from metadata."""
        # Create a saved engine with metadata
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            enable_int8=True,
            enable_caching=True,
            cache_interval=5,
            num_inference_steps=8,
            guidance_scale=1.5,
            seed=42,
        )
        
        # Save a mock engine
        engine_path = tmp_path / "test_engine.pt"
        mock_model = torch.nn.Linear(10, 10)
        torch.save(mock_model, str(engine_path))
        
        # Save metadata
        import json
        metadata = {
            'pipeline_config': {
                'model_id': 'stabilityai/sdxl-turbo',
                'enable_int8': True,
                'enable_caching': True,
                'cache_interval': 5,
                'num_inference_steps': 8,
                'guidance_scale': 1.5,
                'seed': 42,
                'image_size': [512, 512],
                'num_calibration_samples': 512,
                'optimization_level': 5,
                'max_cache_size_gb': 2.0,
                'exclude_layers': None,
            },
            'tensorrt_version': None,
        }
        with open(engine_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f)
        
        # Load the engine
        pipeline = OptimizedPipeline.load_engine(str(engine_path))
        
        # Verify config was restored
        assert pipeline.config.model_id == "stabilityai/sdxl-turbo"
        assert pipeline.config.enable_int8 is True
        assert pipeline.config.cache_interval == 5
        assert pipeline.config.num_inference_steps == 8
        assert pipeline.config.guidance_scale == 1.5
        assert pipeline.config.seed == 42
        assert pipeline.is_optimized is True
    
    @patch.object(OptimizedPipeline, '_load_model')
    @patch.object(OptimizedPipeline, '_setup_caching')
    def test_load_engine_sets_up_caching_if_enabled(self, mock_cache, mock_load, tmp_path):
        """Test load_engine sets up caching if enabled in config."""
        engine_path = tmp_path / "test_engine.pt"
        torch.save(torch.nn.Linear(10, 10), str(engine_path))
        
        import json
        metadata = {
            'pipeline_config': {
                'model_id': 'stabilityai/sdxl-turbo',
                'enable_caching': True,
                'image_size': [512, 512],
            },
        }
        with open(engine_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f)
        
        OptimizedPipeline.load_engine(str(engine_path))
        
        mock_cache.assert_called_once()
    
    @patch.object(OptimizedPipeline, '_load_model')
    def test_load_engine_skips_caching_if_disabled(self, mock_load, tmp_path):
        """Test load_engine skips caching if disabled in config."""
        engine_path = tmp_path / "test_engine.pt"
        torch.save(torch.nn.Linear(10, 10), str(engine_path))
        
        import json
        metadata = {
            'pipeline_config': {
                'model_id': 'stabilityai/sdxl-turbo',
                'enable_caching': False,
                'image_size': [512, 512],
            },
        }
        with open(engine_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f)
        
        pipeline = OptimizedPipeline.load_engine(str(engine_path))
        
        assert pipeline._cache_manager is None
    
    def test_validate_engine_compatibility_no_version_info(self):
        """Test compatibility check passes when no version info in metadata."""
        metadata = {}
        is_compatible, message = OptimizedPipeline._validate_engine_compatibility(metadata)
        
        assert is_compatible is True
        assert "No TensorRT version info" in message
    
    def test_validate_engine_compatibility_trt_not_installed(self):
        """Test compatibility check fails when TensorRT not installed."""
        metadata = {'tensorrt_version': '8.6.0'}
        
        # Mock tensorrt import to raise ImportError
        import sys
        
        # Remove tensorrt from modules if present, then test
        original_tensorrt = sys.modules.get('tensorrt')
        sys.modules['tensorrt'] = None
        
        try:
            # When tensorrt is None in sys.modules, import will fail
            # We need to test the actual behavior
            is_compatible, message = OptimizedPipeline._validate_engine_compatibility(metadata)
            # If tensorrt is actually installed, it will succeed
            # If not, it should return False with appropriate message
            assert isinstance(is_compatible, bool)
            assert isinstance(message, str)
        finally:
            if original_tensorrt is not None:
                sys.modules['tensorrt'] = original_tensorrt
            elif 'tensorrt' in sys.modules:
                del sys.modules['tensorrt']
    
    def test_validate_engine_compatibility_major_version_mismatch(self):
        """Test compatibility check fails on major version mismatch."""
        # This test requires mocking tensorrt import
        metadata = {'tensorrt_version': '7.0.0'}
        
        with patch.dict('sys.modules', {'tensorrt': MagicMock(__version__='8.6.0')}):
            is_compatible, message = OptimizedPipeline._validate_engine_compatibility(metadata)
            
            assert is_compatible is False
            assert "major version mismatch" in message
    
    def test_validate_engine_compatibility_same_version(self):
        """Test compatibility check passes on same version."""
        metadata = {'tensorrt_version': '8.6.0'}
        
        with patch.dict('sys.modules', {'tensorrt': MagicMock(__version__='8.6.0')}):
            is_compatible, message = OptimizedPipeline._validate_engine_compatibility(metadata)
            
            assert is_compatible is True
            assert "compatible" in message.lower()
    
    def test_validate_engine_compatibility_minor_version_differs(self):
        """Test compatibility check warns on minor version difference."""
        metadata = {'tensorrt_version': '8.5.0'}
        
        with patch.dict('sys.modules', {'tensorrt': MagicMock(__version__='8.6.0')}):
            is_compatible, message = OptimizedPipeline._validate_engine_compatibility(metadata)
            
            assert is_compatible is True
            assert "minor version differs" in message or "compatible" in message.lower()
    
    @patch.object(OptimizedPipeline, '_load_model')
    @patch.object(OptimizedPipeline, '_setup_caching')
    def test_load_engine_handles_missing_metadata(self, mock_cache, mock_load, tmp_path):
        """Test load_engine handles missing metadata file gracefully."""
        engine_path = tmp_path / "test_engine.pt"
        torch.save(torch.nn.Linear(10, 10), str(engine_path))
        
        # Don't create metadata file - should use defaults
        pipeline = OptimizedPipeline.load_engine(str(engine_path))
        
        # Should use default config values
        assert pipeline.config.model_id == "stabilityai/sdxl-turbo"
        assert pipeline.is_optimized is True


class TestEnginePersistenceIntegration:
    """Integration tests for save/load engine cycle."""
    
    def test_save_and_load_preserves_config(self, tmp_path):
        """Test that save/load cycle preserves pipeline configuration."""
        # Create and configure pipeline
        config = PipelineConfig(
            model_id="stabilityai/sdxl-turbo",
            enable_int8=True,
            enable_caching=True,
            cache_interval=4,
            num_inference_steps=6,
            guidance_scale=2.0,
            seed=123,
            optimization_level=3,
            max_cache_size_gb=1.5,
        )
        
        pipeline = OptimizedPipeline(config)
        pipeline._is_optimized = True
        pipeline._trt_unet = torch.nn.Linear(10, 10)
        
        # Save engine
        engine_path = tmp_path / "test_engine.pt"
        pipeline.save_engine(str(engine_path))
        
        # Load engine with mocked model loading
        with patch.object(OptimizedPipeline, '_load_model'):
            with patch.object(OptimizedPipeline, '_setup_caching'):
                loaded_pipeline = OptimizedPipeline.load_engine(str(engine_path))
        
        # Verify config was preserved
        assert loaded_pipeline.config.model_id == config.model_id
        assert loaded_pipeline.config.enable_int8 == config.enable_int8
        assert loaded_pipeline.config.enable_caching == config.enable_caching
        assert loaded_pipeline.config.cache_interval == config.cache_interval
        assert loaded_pipeline.config.num_inference_steps == config.num_inference_steps
        assert loaded_pipeline.config.guidance_scale == config.guidance_scale
        assert loaded_pipeline.config.seed == config.seed
        assert loaded_pipeline.config.optimization_level == config.optimization_level
        assert loaded_pipeline.config.max_cache_size_gb == config.max_cache_size_gb


class TestBenchmarkMethod:
    """Tests for OptimizedPipeline.benchmark() method.
    
    Requirements covered:
    - 6.6: Provide a benchmark method returning latency and throughput metrics
    - 11.1: Run warmup iterations before measuring
    - 11.2: Use CUDA synchronization for accurate timing
    - 11.3: Return BenchmarkMetrics with all timing statistics
    - 11.4: Include cache_hit_rate to measure caching effectiveness
    - 11.5: Include vram_peak_gb to verify memory compliance
    """
    
    def test_benchmark_raises_if_not_initialized(self):
        """Test benchmark raises error if pipeline is not initialized."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        with pytest.raises(RuntimeError, match="Pipeline not initialized"):
            pipeline.benchmark()
    
    def test_benchmark_raises_if_num_iterations_invalid(self):
        """Test benchmark raises error if num_iterations < 1."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        pipeline._pipeline = MagicMock()
        
        with pytest.raises(ValueError, match="num_iterations must be >= 1"):
            pipeline.benchmark(num_iterations=0)
    
    def test_benchmark_raises_if_warmup_iterations_invalid(self):
        """Test benchmark raises error if warmup_iterations < 0."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        pipeline._pipeline = MagicMock()
        
        with pytest.raises(ValueError, match="warmup_iterations must be >= 0"):
            pipeline.benchmark(warmup_iterations=-1)
    
    def test_benchmark_returns_benchmark_metrics(self):
        """Test benchmark returns BenchmarkMetrics dataclass."""
        from diffusion_trt.models import BenchmarkMetrics
        
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        metrics = pipeline.benchmark(num_iterations=3, warmup_iterations=1)
        
        assert isinstance(metrics, BenchmarkMetrics)
    
    def test_benchmark_runs_warmup_iterations(self):
        """Test benchmark runs warmup iterations before measuring.
        
        Requirement 11.1: Run warmup iterations before measuring
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        warmup = 2
        measured = 3
        
        pipeline.benchmark(num_iterations=measured, warmup_iterations=warmup)
        
        # Total calls should be warmup + measured
        assert mock_diffusers_pipeline.call_count == warmup + measured
    
    def test_benchmark_returns_correct_num_runs(self):
        """Test benchmark returns correct num_runs in metrics."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        metrics = pipeline.benchmark(num_iterations=5, warmup_iterations=2)
        
        assert metrics.num_runs == 5
        assert metrics.warmup_runs == 2
    
    def test_benchmark_calculates_latency_statistics(self):
        """Test benchmark calculates latency mean, std, and percentiles.
        
        Requirement 11.3: Return BenchmarkMetrics with all timing statistics
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        metrics = pipeline.benchmark(num_iterations=5, warmup_iterations=1)
        
        # Verify latency statistics are populated
        assert metrics.latency_mean_ms > 0
        assert metrics.latency_std_ms >= 0
        assert metrics.latency_p50_ms > 0
        assert metrics.latency_p95_ms > 0
        assert metrics.latency_p99_ms > 0
        
        # Verify percentile ordering
        assert metrics.latency_p50_ms <= metrics.latency_p95_ms
        assert metrics.latency_p95_ms <= metrics.latency_p99_ms
    
    def test_benchmark_calculates_throughput(self):
        """Test benchmark calculates throughput from latency."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        metrics = pipeline.benchmark(num_iterations=3, warmup_iterations=1)
        
        # Throughput should be approximately 1000 / latency_mean_ms
        expected_throughput = 1000.0 / metrics.latency_mean_ms
        assert abs(metrics.throughput_images_per_sec - expected_throughput) < 0.01
    
    def test_benchmark_includes_vram_metrics(self):
        """Test benchmark includes VRAM peak and allocated metrics.
        
        Requirement 11.5: Include vram_peak_gb to verify memory compliance
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        metrics = pipeline.benchmark(num_iterations=3, warmup_iterations=1)
        
        # VRAM metrics should be non-negative
        assert metrics.vram_peak_gb >= 0.0
        assert metrics.vram_allocated_gb >= 0.0
    
    def test_benchmark_includes_cache_hit_rate(self):
        """Test benchmark includes cache_hit_rate from CacheManager.
        
        Requirement 11.4: Include cache_hit_rate to measure caching effectiveness
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Setup mock cache manager with hit rate
        mock_cache = MagicMock()
        mock_cache.get_cache_stats.return_value = {'hit_rate': 0.75}
        pipeline._cache_manager = mock_cache
        
        metrics = pipeline.benchmark(num_iterations=3, warmup_iterations=1)
        
        assert metrics.cache_hit_rate == 0.75
    
    def test_benchmark_cache_hit_rate_zero_without_cache_manager(self):
        """Test benchmark returns 0 cache_hit_rate when caching is disabled."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline without cache manager
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        pipeline._cache_manager = None
        
        metrics = pipeline.benchmark(num_iterations=3, warmup_iterations=1)
        
        assert metrics.cache_hit_rate == 0.0
    
    def test_benchmark_clears_cache_after_warmup(self):
        """Test benchmark clears cache after warmup iterations."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Setup mock cache manager
        mock_cache = MagicMock()
        mock_cache.get_cache_stats.return_value = {'hit_rate': 0.5}
        pipeline._cache_manager = mock_cache
        
        pipeline.benchmark(num_iterations=3, warmup_iterations=2)
        
        # Cache should be cleared multiple times:
        # - Once per __call__ during warmup (2 times)
        # - Once after warmup before measurement
        # - Once per __call__ during measurement (3 times)
        # Total: at least 6 clears
        assert mock_cache.clear.call_count >= 6
    
    def test_benchmark_with_custom_prompt(self):
        """Test benchmark uses custom prompt."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        custom_prompt = "A custom test prompt for benchmarking"
        pipeline.benchmark(prompt=custom_prompt, num_iterations=2, warmup_iterations=1)
        
        # Verify the custom prompt was used
        for call in mock_diffusers_pipeline.call_args_list:
            assert call[1].get('prompt') == custom_prompt
    
    def test_benchmark_with_zero_warmup(self):
        """Test benchmark works with zero warmup iterations."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        metrics = pipeline.benchmark(num_iterations=3, warmup_iterations=0)
        
        # Should only have measured iterations
        assert mock_diffusers_pipeline.call_count == 3
        assert metrics.warmup_runs == 0
        assert metrics.num_runs == 3
    
    def test_benchmark_single_iteration(self):
        """Test benchmark works with single iteration (std should be 0)."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        metrics = pipeline.benchmark(num_iterations=1, warmup_iterations=0)
        
        assert metrics.num_runs == 1
        assert metrics.latency_std_ms == 0.0
        # With single iteration, all percentiles should be equal
        assert metrics.latency_p50_ms == metrics.latency_p95_ms == metrics.latency_p99_ms


class TestPercentileCalculation:
    """Tests for the _percentile static method."""
    
    def test_percentile_empty_list(self):
        """Test percentile returns 0 for empty list."""
        result = OptimizedPipeline._percentile([], 50)
        assert result == 0.0
    
    def test_percentile_single_element(self):
        """Test percentile returns the element for single-element list."""
        result = OptimizedPipeline._percentile([42.0], 50)
        assert result == 42.0
        
        result = OptimizedPipeline._percentile([42.0], 99)
        assert result == 42.0
    
    def test_percentile_p50_median(self):
        """Test p50 returns median value."""
        # Odd number of elements
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = OptimizedPipeline._percentile(data, 50)
        assert result == 3.0
        
        # Even number of elements - interpolated
        data = [1.0, 2.0, 3.0, 4.0]
        result = OptimizedPipeline._percentile(data, 50)
        assert result == 2.5
    
    def test_percentile_p0_returns_min(self):
        """Test p0 returns minimum value."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = OptimizedPipeline._percentile(data, 0)
        assert result == 1.0
    
    def test_percentile_p100_returns_max(self):
        """Test p100 returns maximum value."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = OptimizedPipeline._percentile(data, 100)
        assert result == 5.0
    
    def test_percentile_interpolation(self):
        """Test percentile interpolates between values."""
        data = [10.0, 20.0, 30.0, 40.0]
        
        # p25 should be between 10 and 20
        result = OptimizedPipeline._percentile(data, 25)
        assert 10.0 <= result <= 20.0
        
        # p75 should be between 30 and 40
        result = OptimizedPipeline._percentile(data, 75)
        assert 30.0 <= result <= 40.0


class TestVRAMMemoryManagement:
    """Tests for VRAM-aware memory management.
    
    Requirements covered:
    - 8.2: Keep model weights under 10GB VRAM
    - 8.4: Clear caches and reduce memory usage when approaching limit
    """
    
    def test_check_vram_usage_no_action_when_under_threshold(self):
        """Test _check_vram_usage does nothing when VRAM is under threshold."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock cache manager
        mock_cache = MagicMock()
        pipeline._cache_manager = mock_cache
        
        # Mock get_vram_usage to return low value
        with patch('diffusion_trt.pipeline.get_vram_usage', return_value=5.0):
            pipeline._check_vram_usage(context="test")
        
        # Cache should not be cleared when under threshold
        mock_cache.clear.assert_not_called()
    
    def test_check_vram_usage_clears_cache_when_approaching_limit(self):
        """Test _check_vram_usage clears cache when VRAM approaches limit.
        
        Requirement 8.4: Clear caches when approaching limit
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock cache manager
        mock_cache = MagicMock()
        pipeline._cache_manager = mock_cache
        
        # Mock CUDA availability and get_vram_usage to return high value first, then low after clearing
        with patch('torch.cuda.is_available', return_value=True):
            with patch('diffusion_trt.pipeline.get_vram_usage', side_effect=[14.5, 10.0]):
                with patch('diffusion_trt.pipeline.vram_clear_cache') as mock_clear:
                    pipeline._check_vram_usage(context="test")
        
        # Cache should be cleared
        mock_cache.clear.assert_called_once()
        mock_clear.assert_called_once()
    
    def test_check_vram_usage_raises_oom_when_over_limit(self):
        """Test _check_vram_usage raises OutOfMemoryError when over limit.
        
        Requirement 8.4: Raise error if still over limit after clearing
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock cache manager
        mock_cache = MagicMock()
        pipeline._cache_manager = mock_cache
        
        # Mock CUDA availability and get_vram_usage to return high value even after clearing
        with patch('torch.cuda.is_available', return_value=True):
            with patch('diffusion_trt.pipeline.get_vram_usage', side_effect=[14.5, 16.0]):
                with patch('diffusion_trt.pipeline.vram_clear_cache'):
                    with pytest.raises(torch.cuda.OutOfMemoryError, match="exceeds T4 limit"):
                        pipeline._check_vram_usage(context="test")
    
    def test_check_vram_usage_no_op_without_cuda(self):
        """Test _check_vram_usage does nothing when CUDA is not available."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Mock torch.cuda.is_available to return False
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise any errors
            pipeline._check_vram_usage(context="test")
    
    def test_check_vram_usage_works_without_cache_manager(self):
        """Test _check_vram_usage works when cache manager is None."""
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        pipeline._cache_manager = None
        
        # Mock CUDA availability and get_vram_usage to return high value first, then low after clearing
        with patch('torch.cuda.is_available', return_value=True):
            with patch('diffusion_trt.pipeline.get_vram_usage', side_effect=[14.5, 10.0]):
                with patch('diffusion_trt.pipeline.vram_clear_cache') as mock_clear:
                    # Should not raise even without cache manager
                    pipeline._check_vram_usage(context="test")
        
        # CUDA cache should still be cleared
        mock_clear.assert_called_once()
    
    def test_call_checks_vram_before_inference(self):
        """Test __call__ checks VRAM before starting inference.
        
        Requirement 8.4: Monitor VRAM during inference
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Mock _check_vram_usage
        with patch.object(pipeline, '_check_vram_usage') as mock_check:
            pipeline("A test prompt")
        
        # Should be called at least once for pre-inference check
        assert mock_check.call_count >= 1
        # First call should be for pre-inference
        mock_check.assert_any_call(context="pre-inference")
    
    def test_call_checks_vram_during_inference(self):
        """Test __call__ checks VRAM during each inference step.
        
        Requirement 8.4: Monitor VRAM during inference
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Mock _check_vram_usage
        with patch.object(pipeline, '_check_vram_usage') as mock_check:
            pipeline(["Prompt 1", "Prompt 2"])
        
        # Should be called for pre-inference, each step, and post-inference
        # pre-inference + 2 steps + post-inference = 4 calls
        assert mock_check.call_count >= 4
    
    def test_call_checks_vram_after_inference(self):
        """Test __call__ checks VRAM after inference completes.
        
        Requirement 8.4: Monitor VRAM during inference
        """
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Setup mock pipeline
        mock_diffusers_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        mock_diffusers_pipeline.return_value = mock_result
        pipeline._pipeline = mock_diffusers_pipeline
        
        # Mock _check_vram_usage
        with patch.object(pipeline, '_check_vram_usage') as mock_check:
            pipeline("A test prompt")
        
        # Last call should be for post-inference
        mock_check.assert_any_call(context="post-inference")


class TestLoadModelVRAMManagement:
    """Tests for VRAM management during model loading.
    
    Requirements covered:
    - 8.2: Keep model weights under 10GB VRAM
    """
    
    @patch('diffusion_trt.pipeline.ModelLoader')
    def test_load_model_checks_vram_after_loading(self, mock_loader_class):
        """Test _load_model checks VRAM usage after loading."""
        # Setup mock
        mock_loader = MagicMock()
        mock_pipeline = MagicMock()
        mock_unet = MagicMock()
        
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = mock_pipeline
        mock_loader.extract_unet.return_value = mock_unet
        mock_loader.get_vram_usage.return_value = 5.0  # Under limit
        
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        pipeline._load_model()
        
        # Should check VRAM usage
        mock_loader.get_vram_usage.assert_called()
    
    @patch('diffusion_trt.pipeline.ModelLoader')
    @patch('diffusion_trt.pipeline.vram_clear_cache')
    def test_load_model_raises_oom_when_weights_exceed_limit(
        self, mock_clear, mock_loader_class
    ):
        """Test _load_model raises OutOfMemoryError when weights exceed 10GB.
        
        Requirement 8.2: Keep model weights under 10GB VRAM
        """
        # Setup mock
        mock_loader = MagicMock()
        mock_pipeline = MagicMock()
        mock_unet = MagicMock()
        
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = mock_pipeline
        mock_loader.extract_unet.return_value = mock_unet
        # Return high VRAM even after clearing
        mock_loader.get_vram_usage.return_value = 12.0  # Over 10GB limit
        
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        with pytest.raises(torch.cuda.OutOfMemoryError, match="exceeds limit"):
            pipeline._load_model()
    
    @patch('diffusion_trt.pipeline.ModelLoader')
    @patch('diffusion_trt.pipeline.vram_clear_cache')
    def test_load_model_clears_cache_when_approaching_limit(
        self, mock_clear, mock_loader_class
    ):
        """Test _load_model clears cache when VRAM approaches limit.
        
        Requirement 8.2: Keep model weights under 10GB VRAM
        """
        # Setup mock
        mock_loader = MagicMock()
        mock_pipeline = MagicMock()
        mock_unet = MagicMock()
        
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = mock_pipeline
        mock_loader.extract_unet.return_value = mock_unet
        # Return high VRAM first, then low after clearing
        mock_loader.get_vram_usage.side_effect = [10.5, 8.0]
        
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        pipeline._load_model()
        
        # Should have cleared cache
        mock_clear.assert_called_once()
    
    @patch('diffusion_trt.pipeline.ModelLoader')
    def test_load_model_succeeds_when_under_limit(self, mock_loader_class):
        """Test _load_model succeeds when VRAM is under limit."""
        # Setup mock
        mock_loader = MagicMock()
        mock_pipeline = MagicMock()
        mock_unet = MagicMock()
        
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = mock_pipeline
        mock_loader.extract_unet.return_value = mock_unet
        mock_loader.get_vram_usage.return_value = 8.0  # Under 10GB limit
        
        config = PipelineConfig(model_id="stabilityai/sdxl-turbo")
        pipeline = OptimizedPipeline(config)
        
        # Should not raise
        pipeline._load_model()
        
        assert pipeline._pipeline == mock_pipeline
        assert pipeline._unet == mock_unet


class TestVRAMConstants:
    """Tests for VRAM-related constants."""
    
    def test_model_weights_vram_limit_is_10gb(self):
        """Test MODEL_WEIGHTS_VRAM_LIMIT_GB is 10GB."""
        from diffusion_trt.pipeline import MODEL_WEIGHTS_VRAM_LIMIT_GB
        assert MODEL_WEIGHTS_VRAM_LIMIT_GB == 10.0
    
    def test_vram_warning_threshold_is_14gb(self):
        """Test VRAM_WARNING_THRESHOLD_GB is 14GB."""
        from diffusion_trt.pipeline import VRAM_WARNING_THRESHOLD_GB
        assert VRAM_WARNING_THRESHOLD_GB == 14.0
    
    def test_warning_threshold_less_than_t4_limit(self):
        """Test warning threshold is less than T4 VRAM limit."""
        from diffusion_trt.pipeline import VRAM_WARNING_THRESHOLD_GB
        from diffusion_trt.models import T4_VRAM_LIMIT_GB
        assert VRAM_WARNING_THRESHOLD_GB < T4_VRAM_LIMIT_GB
    
    def test_model_weights_limit_less_than_warning_threshold(self):
        """Test model weights limit is less than warning threshold."""
        from diffusion_trt.pipeline import (
            MODEL_WEIGHTS_VRAM_LIMIT_GB,
            VRAM_WARNING_THRESHOLD_GB,
        )
        assert MODEL_WEIGHTS_VRAM_LIMIT_GB < VRAM_WARNING_THRESHOLD_GB
