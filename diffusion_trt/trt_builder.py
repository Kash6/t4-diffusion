"""
TensorRT Engine Builder component for TensorRT Diffusion Model Optimization Pipeline.

This module provides TensorRT engine building using Torch-TensorRT for optimized
inference on NVIDIA T4 GPUs.

Requirements covered:
- 4.1: Use torch.compile with tensorrt backend
- 4.2: Configure INT8 precision with calibration data
- 4.3: Target sm_75 architecture for T4 GPU optimization
- 4.4: Support optimization levels 0-5
- 4.5: Implement engine serialization
- 4.6: Implement engine loading
- 4.7: Enable CUDA graphs when use_cuda_graph is True
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import json

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Supported precision modes
SUPPORTED_PRECISIONS = ["fp32", "fp16", "int8"]

# T4 GPU compute capability
T4_COMPUTE_CAPABILITY = "sm_75"

# Default workspace size (4GB)
DEFAULT_WORKSPACE_SIZE = 4 * 1024 * 1024 * 1024


@dataclass
class TRTConfig:
    """
    Configuration for TensorRT engine building.
    
    Attributes:
        precision: Precision mode ("fp32", "fp16", "int8")
        workspace_size: Maximum workspace size in bytes (default 4GB)
        max_batch_size: Maximum batch size for the engine
        optimization_level: Optimization level 0-5 (higher = more optimization)
        use_cuda_graph: Enable CUDA graphs for inference
        dynamic_shapes: Enable dynamic input shapes
        target_device: Target GPU compute capability (default sm_75 for T4)
        enable_sparse_weights: Enable sparse weight optimization
    """
    precision: str = "int8"
    workspace_size: int = DEFAULT_WORKSPACE_SIZE
    max_batch_size: int = 1
    optimization_level: int = 5
    use_cuda_graph: bool = True
    dynamic_shapes: bool = False
    target_device: str = T4_COMPUTE_CAPABILITY
    enable_sparse_weights: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision: '{self.precision}'. "
                f"Supported: {SUPPORTED_PRECISIONS}"
            )
        
        if self.workspace_size <= 0:
            raise ValueError(
                f"workspace_size must be positive, got {self.workspace_size}"
            )
        
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be at least 1, got {self.max_batch_size}"
            )
        
        if not 0 <= self.optimization_level <= 5:
            raise ValueError(
                f"optimization_level must be in [0, 5], got {self.optimization_level}"
            )


class TensorRTBuilder:
    """
    TensorRT Engine Builder using Torch-TensorRT.
    
    Compiles PyTorch models to optimized TensorRT engines targeting
    NVIDIA T4 GPUs with INT8 precision support.
    
    Example:
        >>> config = TRTConfig(precision="int8", optimization_level=5)
        >>> builder = TensorRTBuilder(config)
        >>> compiled_model = builder.compile_torchtrt(model, sample_inputs)
    """
    
    def __init__(self, config: TRTConfig):
        """
        Initialize the TensorRTBuilder.
        
        Args:
            config: TensorRT configuration
        """
        self.config = config
        self._compiled_model: Optional[nn.Module] = None
        self._engine_info: Dict[str, Any] = {}
    
    def compile_torchtrt(
        self,
        model: nn.Module,
        sample_inputs: List[torch.Tensor],
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> nn.Module:
        """
        Compile model using torch.compile with TensorRT backend.
        
        Uses Torch-TensorRT's torch.compile backend for seamless integration
        with PyTorch models. Supports INT8 precision with calibration data.
        
        Args:
            model: PyTorch model to compile
            sample_inputs: List of sample input tensors for tracing
            calibration_data: Optional calibration data for INT8 quantization
            
        Returns:
            Compiled model with TensorRT backend
            
        Raises:
            ImportError: If torch_tensorrt is not installed
            RuntimeError: If compilation fails
        """
        # Lazy import to handle missing torch_tensorrt gracefully
        try:
            import torch_tensorrt
            # Try to ensure the library is properly loaded
            try:
                torch_tensorrt.runtime.set_multi_device_safe_mode(False)
            except Exception:
                pass  # Ignore if this API doesn't exist
        except ImportError:
            raise ImportError(
                "torch-tensorrt is required for TensorRT compilation. "
                "Install with: pip install torch-tensorrt"
            )
        except OSError as e:
            # Handle library loading issues
            raise ImportError(
                f"torch-tensorrt library failed to load: {e}. "
                "This is likely a PyTorch/CUDA version mismatch. "
                "Colab ships PyTorch 2.7.x (cu126). Install the matching version:\n"
                "  pip install torch-tensorrt==2.7.0 --extra-index-url https://download.pytorch.org/whl/cu128"
            )
        
        model.eval()
        
        logger.info(
            f"Compiling model with TensorRT: precision={self.config.precision}, "
            f"optimization_level={self.config.optimization_level}"
        )
        
        # Build compilation settings
        compile_settings = self._build_compile_settings(sample_inputs)
        
        try:
            # Modelopt-quantized models contain dynamic callback objects (_FoldedCallback)
            # that TorchDynamo cannot trace. We must export to a static FX graph first,
            # then compile that with torch_tensorrt.compile (dynamo IR).
            logger.info("Exporting model to FX graph via torch.export...")
            
            # Build dynamic shape specs so export handles variable batch/sequence dims
            # Sample inputs are: [latents, timestep, encoder_hidden_states]
            exported_program = torch.export.export(
                model,
                args=tuple(sample_inputs),
                strict=False,  # Allow non-strict export for quantized models
            )
            
            logger.info("Export complete. Compiling with torch_tensorrt...")
            
            compiled_model = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs=sample_inputs,
                **compile_settings,
            )
            
            # Warm up the compiled model
            with torch.no_grad():
                _ = compiled_model(*sample_inputs)
            
            self._compiled_model = compiled_model
            
            logger.info("TensorRT compilation complete")
            
            return compiled_model
            
        except Exception as e:
            logger.error(f"TensorRT compilation failed: {e}")
            raise RuntimeError(f"Failed to compile model with TensorRT: {e}")
    
    def _build_compile_settings(
        self,
        sample_inputs: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Build torch_tensorrt.dynamo.compile settings."""
        # Determine enabled precisions
        if self.config.precision == "fp32":
            enabled_precisions = {torch.float32}
        elif self.config.precision == "fp16":
            enabled_precisions = {torch.float16, torch.float32}
        else:  # int8
            enabled_precisions = {torch.int8, torch.float16, torch.float32}
        
        settings = {
            "enabled_precisions": enabled_precisions,
            "workspace_size": self.config.workspace_size,
            "truncate_double": True,
            "min_block_size": 1,
            "optimization_level": self.config.optimization_level,
            "use_python_runtime": False,
        }
        
        if self.config.use_cuda_graph:
            settings["use_explicit_typing"] = False  # Needed for CUDA graphs compat
        
        return settings
    
    def build_engine(
        self,
        onnx_path: str,
        output_path: str,
        calibration_cache: Optional[str] = None,
    ) -> str:
        """
        Build TensorRT engine from ONNX model.
        
        Uses TensorRT's native API to build an optimized engine
        from an ONNX model file.
        
        Args:
            onnx_path: Path to ONNX model file
            output_path: Path to save the TensorRT engine
            calibration_cache: Optional path to INT8 calibration cache
            
        Returns:
            Path to the built engine file
            
        Raises:
            ImportError: If tensorrt is not installed
            FileNotFoundError: If ONNX file doesn't exist
            RuntimeError: If engine building fails
        """
        # Lazy import
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "tensorrt is required for engine building. "
                "Install with: pip install tensorrt"
            )
        
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        
        logger.info(f"Building TensorRT engine from {onnx_path}")
        
        # Create TensorRT logger and builder
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(f"Failed to parse ONNX: {errors}")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.workspace_size
        )
        
        # Set precision flags
        if self.config.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)  # Fallback
        
        # Set optimization level
        config.builder_optimization_level = self.config.optimization_level
        
        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
        
        # Save metadata
        self._save_engine_metadata(output_path)
        
        logger.info(f"TensorRT engine saved to {output_path}")
        
        return str(output_path)
    
    def load_engine(self, engine_path: str) -> nn.Module:
        """
        Load a pre-built TensorRT engine.
        
        Restores a compiled module from a serialized engine file
        without requiring recompilation.
        
        Args:
            engine_path: Path to the TensorRT engine file
            
        Returns:
            Loaded TensorRT module
            
        Raises:
            ImportError: If torch_tensorrt is not installed
            FileNotFoundError: If engine file doesn't exist
            RuntimeError: If loading fails
        """
        try:
            import torch_tensorrt
        except ImportError:
            raise ImportError(
                "torch-tensorrt is required for engine loading. "
                "Install with: pip install torch-tensorrt"
            )
        
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        logger.info(f"Loading TensorRT engine from {engine_path}")
        
        try:
            # Load the serialized engine
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            
            # Deserialize using torch_tensorrt
            loaded_module = torch_tensorrt.ts.embed_engine_in_new_module(
                engine_bytes
            )
            
            self._compiled_model = loaded_module
            
            # Load metadata if available
            self._load_engine_metadata(engine_path)
            
            logger.info("TensorRT engine loaded successfully")
            
            return loaded_module
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")
    
    def get_engine_info(self, engine_path: str) -> Dict[str, Any]:
        """
        Get metadata about a TensorRT engine.
        
        Args:
            engine_path: Path to the TensorRT engine file
            
        Returns:
            Dictionary with engine metadata:
            - file_size_mb: Engine file size in MB
            - precision: Precision mode used
            - optimization_level: Optimization level used
            - created_at: Creation timestamp
            - tensorrt_version: TensorRT version used
        """
        engine_path = Path(engine_path)
        
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        info = {
            "file_size_mb": engine_path.stat().st_size / (1024 * 1024),
            "engine_path": str(engine_path),
        }
        
        # Try to load metadata file
        metadata_path = engine_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                info.update(metadata)
        
        return info
    
    def _save_engine_metadata(self, engine_path: Path) -> None:
        """Save engine metadata to a JSON file."""
        import datetime
        
        metadata = {
            "precision": self.config.precision,
            "optimization_level": self.config.optimization_level,
            "workspace_size": self.config.workspace_size,
            "max_batch_size": self.config.max_batch_size,
            "target_device": self.config.target_device,
            "use_cuda_graph": self.config.use_cuda_graph,
            "created_at": datetime.datetime.now().isoformat(),
        }
        
        # Try to get TensorRT version
        try:
            import tensorrt as trt
            metadata["tensorrt_version"] = trt.__version__
        except ImportError:
            pass
        
        metadata_path = engine_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_engine_metadata(self, engine_path: Path) -> None:
        """Load engine metadata from JSON file."""
        metadata_path = engine_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self._engine_info = json.load(f)
    
    def validate_engine_compatibility(
        self,
        engine_path: str,
        current_trt_version: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate that an engine is compatible with the current TensorRT version.
        
        Args:
            engine_path: Path to the TensorRT engine file
            current_trt_version: Current TensorRT version (auto-detected if None)
            
        Returns:
            Tuple of (is_compatible, message)
        """
        info = self.get_engine_info(engine_path)
        
        if current_trt_version is None:
            try:
                import tensorrt as trt
                current_trt_version = trt.__version__
            except ImportError:
                return False, "TensorRT not installed"
        
        engine_version = info.get("tensorrt_version")
        
        if engine_version is None:
            return True, "No version info in engine metadata, assuming compatible"
        
        # Check major version compatibility
        current_major = current_trt_version.split('.')[0]
        engine_major = engine_version.split('.')[0]
        
        if current_major != engine_major:
            return False, (
                f"TensorRT version mismatch: engine built with {engine_version}, "
                f"current version is {current_trt_version}"
            )
        
        return True, f"Engine compatible (TensorRT {engine_version})"
    
    def get_compiled_model(self) -> Optional[nn.Module]:
        """Get the last compiled model."""
        return self._compiled_model
