"""Base class for ONNX Runtime inference with GPU/TensorRT support."""

from pathlib import Path
from typing import Union, List

import onnxruntime
from onnxruntime.capi import _pybind_state as C


class ONNXInferenceBase:
    """Base class for ONNX inference with hardware acceleration."""
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        use_gpu: bool = False,
        use_tensorrt: bool = False,
    ):
        """Initialize ONNX Runtime session.
        
        Args:
            model_path: Path to ONNX model file
            use_gpu: Enable CUDA GPU acceleration
            use_tensorrt: Enable TensorRT acceleration (requires TensorRT)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Setup providers (TensorRT > CUDA > CPU)
        providers = self._get_providers(use_gpu, use_tensorrt)
        
        # Create ONNX Runtime session
        self.session = onnxruntime.InferenceSession(
            str(self.model_path),
            None,
            providers=providers
        )
        
        # Cache input/output names
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.output_names = [node.name for node in self.session.get_outputs()]
    
    def _get_providers(self, use_gpu: bool, use_tensorrt: bool) -> List:
        """Get execution providers based on hardware availability.
        
        Priority: TensorRT > CUDA > CPU
        """
        available_providers = C.get_available_providers()
        providers = []
        
        # TensorRT (highest priority if requested and available)
        if use_tensorrt and "TensorrtExecutionProvider" in available_providers:
            providers.append(('TensorrtExecutionProvider', {}))
        
        # CUDA (second priority if requested and available)
        if use_gpu and "CUDAExecutionProvider" in available_providers:
            providers.append((
                'CUDAExecutionProvider',
                {"cudnn_conv_algo_search": "DEFAULT"}
            ))
        
        # CPU (always available as fallback)
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def run(self, input_data: dict) -> List:
        """Run inference on input data.
        
        Args:
            input_data: Dictionary mapping input names to numpy arrays
            
        Returns:
            List of output arrays
        """
        return self.session.run(self.output_names, input_feed=input_data)
    
    def get_input_feed(self, image_array):
        """Create input feed dictionary.
        
        Args:
            image_array: Numpy array or list of arrays
            
        Returns:
            Dictionary mapping input names to arrays
        """
        if len(self.input_names) == 1:
            return {self.input_names[0]: image_array}
        else:
            # For multiple inputs
            return {
                name: image_array[i] 
                for i, name in enumerate(self.input_names)
            }
