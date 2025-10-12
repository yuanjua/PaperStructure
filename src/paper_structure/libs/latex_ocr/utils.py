"""Utilities for LaTeX OCR."""

import io
from pathlib import Path
from typing import Optional

import requests
import tqdm


class ModelDownloader:
    """Download LaTeX OCR models from GitHub releases."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize model downloader.
        
        Args:
            models_dir: Directory to save models (default: libs/latex_ocr/models)
        """
        self.url = "https://github.com/RapidAI/RapidLaTeXOCR/releases/download/v0.0.0"
        
        if models_dir is None:
            self.models_dir = Path(__file__).parent / "models"
        else:
            self.models_dir = Path(models_dir)
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, file_name: str) -> Path:
        """Download a model file if it doesn't exist.
        
        Args:
            file_name: Name of the file to download
            
        Returns:
            Path to the downloaded file
            
        Raises:
            Exception: If download fails
        """
        save_path = self.models_dir / file_name
        
        if save_path.exists():
            return save_path
        
        full_url = f"{self.url}/{file_name}"
        print(f"Downloading {file_name} from {full_url}")
        
        try:
            file_content = self._download_with_progress(full_url, file_name)
            self._save_file(save_path, file_content)
            print(f"Saved to {save_path}")
            return save_path
        except Exception as e:
            raise Exception(f"Failed to download {file_name}: {e}") from e
    
    @staticmethod
    def _download_with_progress(url: str, name: Optional[str] = None) -> bytes:
        """Download file with progress bar.
        
        Args:
            url: URL to download from
            name: Display name for progress bar
            
        Returns:
            Downloaded file content as bytes
        """
        resp = requests.get(url, stream=True, allow_redirects=True)
        resp.raise_for_status()
        
        total = int(resp.headers.get("content-length", 0))
        bio = io.BytesIO()
        
        with tqdm.tqdm(
            desc=name,
            total=total,
            unit="b",
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for chunk in resp.iter_content(chunk_size=65536):
                bar.update(len(chunk))
                bio.write(chunk)
        
        return bio.getvalue()
    
    @staticmethod
    def _save_file(save_path: Path, file_content: bytes):
        """Save file content to disk.
        
        Args:
            save_path: Path to save file
            file_content: File content as bytes
        """
        with open(save_path, "wb") as f:
            f.write(file_content)
