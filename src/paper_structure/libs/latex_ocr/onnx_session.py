"""ONNX Runtime inference session with GPU/TensorRT support."""

import traceback
from pathlib import Path
from typing import List, Union

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from onnxruntime.capi import _pybind_state as C


class OrtInferSession:
    """ONNX Runtime inference session with automatic hardware acceleration.
    
    Supports multiple execution providers with automatic fallback:
    - TensorrtExecutionProvider (best performance on NVIDIA GPUs)
    - CUDAExecutionProvider (good performance on NVIDIA GPUs)
    - CPUExecutionProvider (fallback)
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path], 
        num_threads: int = -1,
        use_gpu: bool = False,
        use_tensorrt: bool = False,
    ):
        """Initialize ONNX Runtime session.
        
        Args:
            model_path: Path to ONNX model file
            num_threads: Number of threads for CPU execution (-1 for auto)
            use_gpu: Enable CUDA GPU acceleration
            use_tensorrt: Enable TensorRT acceleration (requires TensorRT)
        """
        self.verify_exist(model_path)
        self.model_path = model_path
        self.num_threads = num_threads
        
        self._init_sess_opt()
        self._init_providers(use_gpu, use_tensorrt)
        
        try:
            self.session = InferenceSession(
                str(model_path), 
                sess_options=self.sess_opt, 
                providers=self.providers
            )
        except TypeError:
            # Compatible with older onnxruntime versions
            self.session = InferenceSession(
                str(model_path), 
                sess_options=self.sess_opt
            )
    
    def _init_sess_opt(self):
        """Initialize session options."""
        self.sess_opt = SessionOptions()
        self.sess_opt.log_severity_level = 4
        self.sess_opt.enable_cpu_mem_arena = False
        
        if self.num_threads != -1:
            self.sess_opt.intra_op_num_threads = self.num_threads
        
        self.sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    
    def _init_providers(self, use_gpu: bool, use_tensorrt: bool):
        """Initialize execution providers with hardware acceleration support.
        
        Priority: TensorRT > CUDA > CPU
        """
        available_providers = C.get_available_providers()
        
        self.providers = []
        provider_options = []
        
        # TensorRT (highest priority if requested and available)
        if use_tensorrt and "TensorrtExecutionProvider" in available_providers:
            self.providers.append("TensorrtExecutionProvider")
            provider_options.append({})
        
        # CUDA (second priority if requested and available)
        if use_gpu and "CUDAExecutionProvider" in available_providers:
            self.providers.append("CUDAExecutionProvider")
            provider_options.append({})
        
        # CPU (always available as fallback)
        self.providers.append("CPUExecutionProvider")
        provider_options.append({
            "arena_extend_strategy": "kSameAsRequested",
        })
        
        # Convert to list of tuples format
        self.providers = [
            (provider, options) 
            for provider, options in zip(self.providers, provider_options)
        ]
    
    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        """Run inference on input data.
        
        Args:
            input_content: List of numpy arrays as model inputs
            
        Returns:
            Model output as numpy array
        """
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e
    
    def get_input_names(self) -> List[str]:
        """Get model input names."""
        return [v.name for v in self.session.get_inputs()]
    
    def get_output_name(self, output_idx: int = 0) -> str:
        """Get model output name by index."""
        return self.session.get_outputs()[output_idx].name
    
    def get_metadata(self) -> dict:
        """Get model metadata."""
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict
    
    @staticmethod
    def verify_exist(model_path: Union[Path, str]):
        """Verify that model file exists."""
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist!")
        
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} must be a file")


class ONNXRuntimeError(Exception):
    """Exception raised when ONNX Runtime encounters an error."""
    pass
