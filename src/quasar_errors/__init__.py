import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "analysis": ["Analyser"],
    "bootstrapping": [
        "BaseBootstrapper",
        "SpectrumBootstrapper",
        "StandardBootstrapper",
    ],
    "error_result.error_result": ["ErrorResult"],
    "error_result.error_result_builder": ["ErrorResultBuilder"],
    "model_samples.balmer": ["BalmerSample"],
    "model_samples.continuum": ["PowerLawSample"],
    "model_samples.host": ["HostGalaxySample"],
    "model_samples.iron": [
        "IronSample", 
        "IronSampleList",
    ],
    "model_samples.line": [
        "GaussianSample", 
        "GaussianSampleList",
    ],
    "model_samples.sample": ["Sample"],
    "nested_sampling": [
        "BaseNestedSampler", 
        "UniformNestedSampler",
    ],
    "measure_result": [
        "MeasureResult", 
        "FrozenMeasureResult", 
        "MeasureResultDict",
    ],
}

if TYPE_CHECKING:
    from .analysis import Analyser  # noqa: F401
    from .bootstrapping import (  # noqa: F401
        BaseBootstrapper,
        SpectrumBootstrapper,
        StandardBootstrapper,
    )
    from .error_result.error_result import ErrorResult  # noqa: F401
    from .error_result.error_result_builder import (
        ErrorResultBuilder,  # noqa: F401
    )
    from .measure_result import (  # noqa: F401
        FrozenMeasureResult,
        MeasureResult,
        MeasureResultDict,
    )
    from .model_samples.balmer import BalmerSample  # noqa: F401
    from .model_samples.continuum import PowerLawSample  # noqa: F401
    from .model_samples.host import HostGalaxySample  # noqa: F401
    from .model_samples.iron import IronSample, IronSampleList  # noqa: F401
    from .model_samples.line import (  # noqa: F401
        GaussianSample,
        GaussianSampleList,
    )
    from .model_samples.sample import Sample  # noqa: F401
    from .nested_sampling import (  # noqa: F401
        BaseNestedSampler,
        UniformNestedSampler,
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )