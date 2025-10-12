"""Model factory for YOLOX layout detection."""

import os
import threading
from typing import Optional

from .yolox_model import MODEL_TYPES, UnstructuredYoloXModel


DEFAULT_MODEL = "yolox"


class Models:
    """Singleton class to manage model instances."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """return an instance if one already exists otherwise create an instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Models, cls).__new__(cls)
                    cls._instance.models = {}
        return cls._instance

    def __contains__(self, key):
        return key in self.models

    def __getitem__(self, key: str):
        return self.models.__getitem__(key)

    def __setitem__(self, key: str, value: UnstructuredYoloXModel):
        self.models[key] = value


models: Models = Models()


def get_model(model_name: Optional[str] = None) -> UnstructuredYoloXModel:
    """Gets the model object by model name.
    
    Args:
        model_name: Name of the YOLOX model variant (yolox, yolox_tiny, yolox_quantized)
                   If None, uses default model or one from environment variable.
    
    Returns:
        UnstructuredYoloXModel: Initialized YOLOX model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name is None:
        default_name_from_env = os.environ.get("UNSTRUCTURED_DEFAULT_MODEL_NAME")
        model_name = default_name_from_env if default_name_from_env is not None else DEFAULT_MODEL

    if model_name in models:
        return models[model_name]

    if model_name not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_name}. Available models: {list(MODEL_TYPES.keys())}")

    initialize_params = MODEL_TYPES[model_name]
    
    # Force evaluation of LazyDict by accessing all items
    evaluated_params = {k: initialize_params[k] for k in initialize_params.keys()}
    
    model = UnstructuredYoloXModel()
    model.initialize(**evaluated_params)
    models[model_name] = model
    
    return model
