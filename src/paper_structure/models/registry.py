"""
Unified model registry: download, cache, and resolve paths for all model weights.

Every model is fetched from a single HuggingFace repository via huggingface_hub,
which handles caching, resumable downloads, and integrity checks.

Usage:
    from paper_structure.models import registry

    path = registry.get("latex_ocr", "decoder")   # download + resolve
    registry.ensure_all()                          # pre-download everything
    print(registry.status())                       # show what's cached
"""

from pathlib import Path
from typing import Dict, Optional

from .config import ALL_GROUPS, HF_REPO, ModelFile, ModelGroup


class ModelRegistry:
    """Central manager for all model weights."""

    def __init__(self, repo_id: str = HF_REPO):
        self._repo_id = repo_id
        self._groups = ALL_GROUPS

    @property
    def repo_id(self) -> str:
        return self._repo_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, group_name: str, file_key: str) -> Path:
        """Return the local path for a model file, downloading if needed.

        Args:
            group_name: e.g. "latex_ocr", "yolox", "paddle_ocr"
            file_key:   e.g. "decoder", "yolox", "detector"

        Returns:
            Resolved Path to the model file on disk.
        """
        group = self._resolve_group(group_name)
        mf = self._resolve_file(group, file_key)
        return self._ensure_file(mf)

    def get_group_paths(self, group_name: str) -> Dict[str, Path]:
        """Return *all* resolved paths for a model group.

        Returns:
            Dict mapping file_key -> local Path.
        """
        group = self._resolve_group(group_name)
        return {key: self._ensure_file(mf) for key, mf in group.files.items()}

    def ensure_group(self, group_name: str) -> None:
        """Download all files for a group if not already present."""
        self.get_group_paths(group_name)

    def ensure_all(self) -> None:
        """Download every model file in the project."""
        for name in self._groups:
            self.ensure_group(name)

    def status(self) -> str:
        """Return a human-readable status report."""
        lines = [
            "Model Registry Status",
            f"Repository: {self._repo_id}",
            "=" * 60,
        ]
        for gname, group in self._groups.items():
            lines.append(f"\n{group.name}  ({group.description})")
            for key, mf in group.files.items():
                cached = self._find_cached(mf)
                if cached is not None:
                    mark = "OK"
                    loc = str(cached)
                else:
                    mark = "MISSING"
                    loc = f"hf://{self._repo_id}/{mf.filename}"
                lines.append(f"  [{mark:>7}]  {key:<16} {mf.filename:<35} {loc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_group(self, name: str) -> ModelGroup:
        if name not in self._groups:
            available = ", ".join(self._groups)
            raise KeyError(f"Unknown model group '{name}'. Available: {available}")
        return self._groups[name]

    @staticmethod
    def _resolve_file(group: ModelGroup, key: str) -> ModelFile:
        if key not in group.files:
            available = ", ".join(group.files)
            raise KeyError(
                f"Unknown file '{key}' in group '{group.name}'. Available: {available}"
            )
        return group.files[key]

    def _ensure_file(self, mf: ModelFile) -> Path:
        """Return the local path, downloading via HF Hub if needed."""
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(self._repo_id, mf.filename)
        return Path(local)

    def _find_cached(self, mf: ModelFile) -> Optional[Path]:
        """Check if a file is already in the HuggingFace cache."""
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(self._repo_id, mf.filename)
            if result is not None and isinstance(result, str):
                return Path(result)
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
registry = ModelRegistry()
