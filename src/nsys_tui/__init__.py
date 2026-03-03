"""nsight — Python library for Nsight Systems profile analysis and visualization."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("nsys-ai")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0+dev"

from . import export, export_flat, overlap, profile, projection, search, summary, tree, web

__all__ = [
    "__version__",
    "profile", "projection", "export", "tree", "summary",
    "overlap", "search", "export_flat", "web",
]
