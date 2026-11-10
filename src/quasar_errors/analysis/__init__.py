import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "analyser": ["Analyser"],
    "paqs": [
        "PAQSContinuumOutput",
        "PAQSLineOutput",
        "PAQSOutput",
    ]
}

if TYPE_CHECKING:
    from .analyser import Analyser  # noqa: F401
    from .paqs import (  # noqa: F401
        PAQSContinuumOutput,
        PAQSLineOutput,
        PAQSOutput,
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )