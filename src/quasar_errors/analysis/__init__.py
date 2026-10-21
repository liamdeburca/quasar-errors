import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "analyser": ["Analyser"],
    "measure_result": ["FrozenMeasureResult", "MeasureResult"],
}

if TYPE_CHECKING:
    from .analyser import Analyser  # noqa: F401
    from .measure_result import (  # noqa: F401
        FrozenMeasureResult,
        MeasureResult,
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )