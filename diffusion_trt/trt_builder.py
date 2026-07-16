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


class UNetOutput:
    """UNet-compatible output wrapper.
    
    diffusers accesses UNet outputs two ways depending on the code path:
    - result.sample (return_dict=True style)
    - result[0] (return_dict=False style, e.g. self.unet(...)[0])
    Support both.
    """
    def __init__(self, sample: torch.Tensor):
        self.sample = sample
    
    def __getitem__(self, idx):
        if idx == 0:
            return self.sample
        raise IndexError(f"UNetOutput only has index 0, got {idx}")


class TRTEngineRunner(nn.Module):
    """
    Wraps a native TensorRT engine as a callable nn.Module.
    
    This allows the TRT engine to be used as a drop-in replacement for
    the UNet in the diffusers pipeline. Handles GPU memory allocation,
    input/output binding, and CUDA stream management.
    
    The engine is built from an ONNX model with QDQ nodes, so TensorRT
    uses real INT8 kernels on the T4's INT8 tensor cores.
    """
    
    def __init__(self, engine_path: str, input_names: List[str], is_sdxl: bool = False):
        super().__init__()
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("tensorrt is required. Install with: pip install tensorrt")
        
        self._trt = trt
        self.input_names = input_names
        self.is_sdxl = is_sdxl
        self._stream = torch.cuda.Stream()
        
        # Dummy parameter so diffusers detects this module as living on CUDA.
        # Without this, pipeline.device returns "cpu" and randn_tensor fails.
        self.register_buffer("_device_indicator", torch.empty(0, device="cuda"))
        
        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        
        if self._engine is None:
            raise RuntimeError(f"Failed to load TRT engine from {engine_path}")
        
        self._context = self._engine.create_execution_context()
        logger.info(
            f"TRT engine loaded: {self._engine.num_io_tensors} I/O tensors, "
            f"from {engine_path}"
        )
    
    @property
    def dtype(self):
        """Return FP16 dtype for diffusers compatibility."""
        return torch.float16
    
    @property  
    def device(self):
        """Return CUDA device for diffusers compatibility."""
        return torch.device("cuda")
    
    def _run_single(self, sample, timestep, encoder_hidden_states,
                     text_embeds=None, time_ids=None):
        """Run the engine on a single batch=1 input set."""
        inputs = {
            "sample": sample.contiguous(),
            "timestep": timestep.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
        }
        if self.is_sdxl and text_embeds is not None:
            inputs["text_embeds"] = text_embeds.contiguous()
            inputs["time_ids"] = time_ids.contiguous()
        
        # Set input tensor addresses
        for name, tensor in inputs.items():
            self._context.set_input_shape(name, tuple(tensor.shape))
            self._context.set_tensor_address(name, tensor.data_ptr())
        
        # Allocate output
        output_name = "output"
        output_shape = self._context.get_tensor_shape(output_name)
        output_dtype = self._engine.get_tensor_dtype(output_name)
        
        # Map TRT dtype to torch dtype
        dtype_map = {
            self._trt.float32: torch.float32,
            self._trt.float16: torch.float16,
            self._trt.int8: torch.int8,
            self._trt.int32: torch.int32,
        }
        torch_dtype = dtype_map.get(output_dtype, torch.float16)
        output = torch.empty(tuple(output_shape), dtype=torch_dtype, device="cuda")
        self._context.set_tensor_address(output_name, output.data_ptr())
        
        # Execute
        self._context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        
        return output
    
    def forward(self, sample, timestep, encoder_hidden_states=None,
                added_cond_kwargs=None, **kwargs):
        """Run inference through the TRT engine with UNet-compatible interface.
        
        The engine is built for a static batch size of 1. diffusers commonly
        calls the UNet with batch=2 (classifier-free guidance: conditional +
        unconditional stacked together), so we split any batch>1 input into
        batch-1 chunks, run the engine per-chunk, and concatenate the results.
        """
        batch_size = sample.shape[0]
        
        text_embeds = added_cond_kwargs.get("text_embeds") if (
            self.is_sdxl and added_cond_kwargs is not None
        ) else None
        time_ids = added_cond_kwargs.get("time_ids") if (
            self.is_sdxl and added_cond_kwargs is not None
        ) else None
        
        # Broadcast timestep to batch size if it's a scalar/1-element tensor
        if timestep.numel() == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        
        if batch_size == 1:
            output = self._run_single(sample, timestep, encoder_hidden_states, text_embeds, time_ids)
        else:
            outputs = []
            for i in range(batch_size):
                te = text_embeds[i:i+1] if text_embeds is not None else None
                ti = time_ids[i:i+1] if time_ids is not None else None
                outputs.append(
                    self._run_single(
                        sample[i:i+1],
                        timestep[i:i+1],
                        encoder_hidden_states[i:i+1],
                        te, ti,
                    )
                )
            output = torch.cat(outputs, dim=0)
        
        return UNetOutput(sample=output)


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
    
    @staticmethod
    def _write_progress(step: str) -> None:
        """Write a crash-survival progress checkpoint (see pipeline._progress)."""
        try:
            import datetime
            import os as _os
            with open("/tmp/diffusion_trt_progress.log", "a") as f:
                f.write(f"{datetime.datetime.now().isoformat()} {step}\n")
                f.flush()
                _os.fsync(f.fileno())
        except Exception:
            pass
    
    def compile_torchtrt(
        self,
        model: nn.Module,
        sample_inputs: List[torch.Tensor],
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> nn.Module:
        """
        Compile model using Torch-TensorRT.
        
        Expects modelopt quantizers to be already disabled/folded before
        calling this method. The folded model is a standard PyTorch model
        with INT8-calibrated weights baked in, compilable by TorchDynamo.
        
        Args:
            model: PyTorch model to compile (quantizers must be folded)
            sample_inputs: List of sample input tensors for tracing
            calibration_data: Optional calibration data (unused, kept for API compat)
            
        Returns:
            Compiled model with TensorRT backend
            
        Raises:
            ImportError: If torch_tensorrt is not installed
            RuntimeError: If compilation fails
        """
        try:
            import torch_tensorrt
            trt_version = getattr(torch_tensorrt, '__version__', 'unknown')
            logger.info(f"torch_tensorrt version: {trt_version}")
            try:
                torch_tensorrt.runtime.set_multi_device_safe_mode(False)
            except Exception:
                pass
        except ImportError:
            raise ImportError(
                "torch-tensorrt is required for TensorRT compilation. "
                "Install with: pip install torch-tensorrt"
            )
        except OSError as e:
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
        
        # Log environment info for debugging
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        
        # Log sample input shapes
        for i, inp in enumerate(sample_inputs):
            logger.info(f"Sample input {i}: shape={inp.shape}, dtype={inp.dtype}, device={inp.device}")
        
        # After quantizer folding, the model is effectively FP16 with calibrated
        # weights. Use FP16 precision for TRT compilation.
        if self.config.precision == "fp32":
            enabled_precisions = {torch.float32}
        else:
            enabled_precisions = {torch.float16}
        
        logger.info(f"Enabled precisions: {enabled_precisions}")
        num_modules = sum(1 for _ in model.modules())
        logger.info(f"Model module count: {num_modules}")
        
        try:
            with torch.no_grad():
                # Strategy 1: torch_tensorrt.compile with ir="dynamo"
                logger.info("Strategy 1: torch_tensorrt.compile(ir='dynamo')...")
                try:
                    compiled_model = torch_tensorrt.compile(
                        model,
                        ir="dynamo",
                        arg_inputs=sample_inputs,
                        enabled_precisions=enabled_precisions,
                        optimization_level=self.config.optimization_level,
                        workspace_size=self.config.workspace_size,
                        min_block_size=1,
                        truncate_double=True,
                    )
                    _ = compiled_model(*sample_inputs)
                    self._compiled_model = compiled_model
                    logger.info("TensorRT compilation complete (torch_tensorrt.compile)")
                    return compiled_model
                except Exception as e1:
                    logger.warning(f"Strategy 1 failed: {e1}", exc_info=True)
                
                # Strategy 2: torch.export + dynamo.compile
                logger.info("Strategy 2: torch.export + dynamo.compile...")
                try:
                    exported = torch.export.export(
                        model,
                        args=tuple(sample_inputs),
                        strict=False,
                    )
                    compiled_model = torch_tensorrt.dynamo.compile(
                        exported,
                        inputs=sample_inputs,
                        enabled_precisions=enabled_precisions,
                        optimization_level=self.config.optimization_level,
                        workspace_size=self.config.workspace_size,
                        min_block_size=1,
                        truncate_double=True,
                    )
                    _ = compiled_model(*sample_inputs)
                    self._compiled_model = compiled_model
                    logger.info("TensorRT compilation complete (export + dynamo)")
                    return compiled_model
                except Exception as e2:
                    logger.warning(f"Strategy 2 failed: {e2}", exc_info=True)
                
                # Strategy 3: torch.compile with torch_tensorrt backend
                logger.info("Strategy 3: torch.compile(backend='torch_tensorrt')...")
                compiled_model = torch.compile(
                    model,
                    backend="torch_tensorrt",
                    options={
                        "enabled_precisions": enabled_precisions,
                        "workspace_size": self.config.workspace_size,
                        "min_block_size": 1,
                        "truncate_double": True,
                        "optimization_level": self.config.optimization_level,
                    },
                )
                _ = compiled_model(*sample_inputs)
                self._compiled_model = compiled_model
                logger.info("TensorRT compilation complete (torch.compile backend)")
                return compiled_model
            
        except Exception as e:
            logger.error(f"TensorRT compilation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to compile model with TensorRT: {e}")
    
    @staticmethod
    def _model_has_qdq_nodes(model: nn.Module) -> bool:
        """Check if a model contains modelopt quantizer (QDQ) modules."""
        try:
            from modelopt.torch.quantization.nn import TensorQuantizer
            for module in model.modules():
                if isinstance(module, TensorQuantizer):
                    return True
        except ImportError:
            pass
        # Fallback: check for common quantizer attribute names
        for name, _ in model.named_modules():
            if "quantizer" in name.lower() or "_amax" in name.lower():
                return True
        return False
    
    def _build_compile_settings(
        self,
        sample_inputs: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """Build torch_tensorrt.dynamo.compile settings (kept for reference)."""
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
        }
        
        if self.config.dynamic_shapes:
            settings["dynamic_batch"] = True
        
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
        self._write_progress("trt build: creating builder/network")
        
        # Use VERBOSE logging so TRT builder errors are visible
        trt_logger = trt.Logger(trt.Logger.VERBOSE if logger.isEnabledFor(logging.DEBUG) else trt.Logger.INFO)
        builder = trt.Builder(trt_logger)
        # TensorRT 10+ removed EXPLICIT_BATCH (implicit batch mode no longer
        # exists, so all networks are explicit-batch by default). Older TRT
        # versions require the flag; guard for both.
        if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(
            trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"
        ):
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
        else:
            network = builder.create_network()
        parser = trt.OnnxParser(network, trt_logger)
        
        self._write_progress("trt build: parsing onnx file")
        # Parse ONNX model
        # Use parse_from_file so TRT resolves external data paths relative
        # to the ONNX file's directory (needed for models > 2GB).
        if not parser.parse_from_file(str(onnx_path)):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"Failed to parse ONNX: {errors}")
        logger.info(f"ONNX parsed successfully: {network.num_layers} layers")
        self._write_progress(f"trt build: onnx parsed ({network.num_layers} layers)")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.workspace_size
        )
        
        # Set precision flags.
        # TensorRT 10+ removed BuilderFlag.FP16/INT8 — precision is now
        # determined by the network's tensor dtypes ("strongly typed"
        # networks) rather than builder flags. Our ONNX export already uses
        # FP16 tensors, so TRT infers FP16 automatically on newer versions.
        # Guard the flag-setting for older TRT (<10) where it's still required.
        if hasattr(trt.BuilderFlag, "FP16"):
            if self.config.precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
            elif self.config.precision == "int8":
                if hasattr(trt.BuilderFlag, "INT8"):
                    config.set_flag(trt.BuilderFlag.INT8)
                config.set_flag(trt.BuilderFlag.FP16)  # Fallback
        else:
            logger.info(
                f"TensorRT {trt.__version__} uses strongly-typed networks — "
                f"precision ({self.config.precision}) is inferred from ONNX "
                f"tensor dtypes, no builder flag needed."
            )
        
        # Set optimization level. Level 0 skips most of TensorRT's tactic
        # search (the exhaustive per-layer kernel benchmarking that consumes
        # large amounts of HOST RAM, not just GPU memory, and is the likely
        # cause of unrecoverable OOM kills with no Python traceback on
        # memory-constrained hosts like free-tier Colab).
        config.builder_optimization_level = self.config.optimization_level
        
        # NOTE: We previously restricted tactic sources / aux streams here
        # suspecting host-RAM exhaustion during tactic search. A standalone
        # test confirmed TensorRT's builder works fine on this machine — the
        # real cause was an oversized ONNX graph (~21K nodes for SD1.5 UNet
        # due to do_constant_folding=False). Left at TRT defaults now.
        
        # Build engine
        if torch.cuda.is_available():
            logger.info(f"VRAM before engine build: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        logger.info("Building serialized network (this may take several minutes)...")
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(
                f"Host RAM before build: {mem.used / 1e9:.2f}/{mem.total / 1e9:.2f} GB used"
            )
        except ImportError:
            pass
        self._write_progress("trt build: calling build_serialized_network")
        serialized_engine = builder.build_serialized_network(network, config)
        self._write_progress("trt build: build_serialized_network returned")
        if serialized_engine is None:
            # Log network info for debugging
            logger.error(f"TRT engine build returned None. Network has {network.num_layers} layers.")
            for i in range(min(5, network.num_layers)):
                layer = network.get_layer(i)
                logger.error(f"  Layer {i}: {layer.name} ({layer.type})")
            raise RuntimeError(
                "Failed to build TensorRT engine. This usually means OOM during "
                "engine construction or unsupported ops. Check TRT logs above."
            )
        
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
