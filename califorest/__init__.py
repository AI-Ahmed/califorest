"""
CaliForest package
"""

# Version from pyproject.toml
from importlib.metadata import version
__version__ = version("califorest")

from .cal_tree import CalibratedTree
from .cal_random_forest import CalibratedForest

from .metrics import hosmer_lemeshow
from .metrics import reliability
from .metrics import spiegelhalter
from .metrics import scaled_brier_score


__all__ = [
    "__version__",
    "CalibratedTree",
    "CalibratedForest",
    "hosmer_lemeshow",
    "reliability",
    "spiegelhalter",
    "scaled_brier_score"
]
