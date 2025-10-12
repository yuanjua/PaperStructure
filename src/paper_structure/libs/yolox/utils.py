"""Utility functions for YOLOX model loading."""

import os
from typing import Any, Callable


class LazyEvaluateInfo:
    """Class that stores the information needed to lazily evaluate a function with given arguments.
    The object stores the information needed for evaluation as a function and its arguments.
    """

    def __init__(self, evaluate: Callable, *args, **kwargs):
        self.evaluate = evaluate
        self.info = (args, kwargs)


class LazyDict(dict):
    """Class that wraps a dict and only evaluates keys of the dict when the key is accessed. 
    Keys that should be evaluated lazily should use LazyEvaluateInfo objects as values. 
    By default when a value is computed from a LazyEvaluateInfo object, it is converted to 
    the raw value in the internal dict, so subsequent accessing of the key will produce the 
    same value.
    """

    def __init__(self, *args, cache=True, **kwargs):
        self.cache = cache
        super().__init__(*args, **kwargs)

    def __getitem__(self, key: Any) -> Any:
        value = super().__getitem__(key)
        if isinstance(value, LazyEvaluateInfo):
            evaluate = value.evaluate
            args, kwargs = value.info
            value = evaluate(*args, **kwargs)
            if self.cache:
                super().__setitem__(key, value)
        return value
    
    def keys(self):
        """Return keys from the underlying dict"""
        return super().keys()
    
    def values(self):
        """Return evaluated values"""
        return [self[key] for key in self.keys()]
    
    def items(self):
        """Return key-value pairs with evaluated values"""
        return [(key, self[key]) for key in self.keys()]


def download_if_needed_and_get_local_path(path_or_repo: str, filename: str, **kwargs) -> str:
    """Returns path to local file if it exists, otherwise treats it as a huggingface repo and
    attempts to download."""
    from huggingface_hub import hf_hub_download
    
    full_path = os.path.join(path_or_repo, filename)
    if os.path.exists(full_path):
        return full_path
    else:
        return hf_hub_download(path_or_repo, filename, **kwargs)
