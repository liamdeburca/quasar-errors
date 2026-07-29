from collections.abc import Iterable
from typing import Self

from pydantic.dataclasses import dataclass
from quasar_models.iron import IronModel
from quasar_typing.astropy import Model_
from quasar_utils.decorators import validate_call

from ..utils.base_sample_list import BaseSampleList
from .iron_sample import IronSample


@dataclass(frozen=True)
class IronSampleList(BaseSampleList[IronSample]):
    samples: list[IronSample]

    @classmethod
    @validate_call
    def fromIronSamples(
        cls,
        samples: Iterable[IronSample],
    ) -> Self:
        return IronSampleList(list(samples))

    @classmethod
    @validate_call
    def fromIronSubmodels(
        cls,
        submodels: Iterable[IronModel],
    ) -> Self:
        return cls.fromIronSamples(
            IronSample.fromIronModel.__wrapped__(IronSample, model)
            for model in submodels
        )

    @classmethod
    @validate_call
    def fromIronModels(cls, model: Model_) -> Self:
        return cls.fromIronSubmodels.__wrapped__(
            cls,
            filter(
                lambda m: isinstance(m, IronModel),
                (model,) if model.n_submodels == 1 else model,
            ),
        )
