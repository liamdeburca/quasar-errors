from collections.abc import Iterable
from typing import Self

from pydantic import validate_call
from pydantic.dataclasses import dataclass
from quasar_models.continuum import PowerLawModel
from quasar_typing.astropy import Model_

from ..utils.base_sample_list import BaseSampleList
from .powerlaw_sample import PowerLawSample


@dataclass(frozen=True)
class PowerLawSampleList(BaseSampleList[PowerLawSample]):
    samples: list[PowerLawSample]

    @classmethod
    @validate_call
    def fromPowerLawSamples(
        cls,
        samples: Iterable[PowerLawSample],
    ) -> Self:
        return PowerLawSampleList(list(samples))

    @classmethod
    @validate_call
    def fromPowerLawSubmodels(
        cls,
        submodels: Iterable[PowerLawModel],
    ) -> Self:
        return cls.fromPowerLawSamples(
            PowerLawSample.fromPowerLawModel.__wrapped__(PowerLawSample, model)
            for model in submodels
        )

    @classmethod
    @validate_call
    def fromPowerLawModels(cls, model: Model_) -> Self:
        return cls.fromPowerLawSubmodels.__wrapped__(
            cls,
            filter(
                lambda m: isinstance(m, PowerLawModel),
                (model,) if model.n_submodels == 1 else model,
            ),
        )
