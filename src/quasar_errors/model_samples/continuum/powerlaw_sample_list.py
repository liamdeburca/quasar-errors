raise DeprecationWarning("PowerLawSampleList is deprecated.")


from collections.abc import Iterable
from typing import Self

import pydantic
from quasar_models.continuum import PowerLawModel
from quasar_typing.astropy import ModelType

from ..utils.base_sample_list import BaseSampleList
from .powerlaw_sample import PowerLawSample


@pydantic.dataclasses.dataclass(frozen=True)
class PowerLawSampleList(BaseSampleList[PowerLawSample]):
    samples: list[PowerLawSample]

    @classmethod
    @pydantic.validate_call
    def fromPowerLawSamples(
        cls,
        samples: Iterable[PowerLawSample],
    ) -> Self:
        return PowerLawSampleList(list(samples))

    @classmethod
    @pydantic.validate_call
    def fromPowerLawSubmodels(
        cls,
        submodels: Iterable[PowerLawModel],
    ) -> Self:
        return cls.fromPowerLawSamples(
            PowerLawSample.fromPowerLawModel(model)
            for model in submodels
        )

    @classmethod
    @pydantic.validate_call
    def fromPowerLawModels(cls, model: ModelType) -> Self:
        return cls.fromPowerLawSubmodels(filter(
            lambda m: isinstance(m, PowerLawModel),
            (model,) if model.n_submodels == 1 else model,
        ))
