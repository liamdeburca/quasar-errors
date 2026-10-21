import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "error_result": ["ErrorResult"],
    "error_result_builder": ["ErrorResultBuilder"],
}

if TYPE_CHECKING:
    from .error_result import ErrorResult  # noqa: F401
    from .error_result_builder import ErrorResultBuilder  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )