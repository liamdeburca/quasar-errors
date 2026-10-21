import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "balmer": ["BalmerSample"],
    "continuum": ["PowerLawSample"],
    "host": ["HostGalaxySample"],
    "iron": ["IronSample", "IronSampleList"],
    "line": ["GaussianSample", "GaussianSampleList"],
    "sample": ["Sample"],
}

if TYPE_CHECKING:
    from .balmer import BalmerSample  # noqa: F401
    from .continuum import PowerLawSample  # noqa: F401
    from .host import HostGalaxySample  # noqa: F401
    from .iron import IronSample, IronSampleList  # noqa: F401
    from .line import GaussianSample, GaussianSampleList  # noqa: F401
    from .sample import Sample  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )