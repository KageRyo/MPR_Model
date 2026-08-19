"""Public Python API for WQSurrogateModels."""

from .wqi import categorize_score, direct_wqi5_score
from .version import __version__

__all__ = ["__version__", "categorize_score", "direct_wqi5_score"]
