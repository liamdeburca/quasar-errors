from collections.abc import Iterable
from typing import Self

import pydantic
from quasar_models.iron import IronModel
from quasar_typing.astropy import ModelType

from ..utils.base_sample_list import BaseSampleList
from .iron_sample import IronSample


class IronSampleList(BaseSampleList[IronSample]):
    samples: list[IronSample]


    @classmethod
    @pydantic.validate_call
    def fromIronSamples(
        cls,
        samples: Iterable[IronSample],
    ) -> Self:
        return cls(samples=sorted(samples, key=lambda s: s.sorting_key))

    @classmethod
    @pydantic.validate_call
    def fromIronSubmodels(
        cls,
        submodels: Iterable[IronModel],
    ) -> Self:
        return cls.fromIronSamples(
            IronSample.fromIronModel(model)
            for model in sorted(submodels, key=lambda m: m.sorting_key)
        )

    @classmethod
    @pydantic.validate_call
    def fromIronModels(cls, model: ModelType) -> Self:
        return cls.fromIronSubmodels(
            filter(
                lambda m: isinstance(m, IronModel),
                (model,) if model.n_submodels == 1 else model,
            ),
        )
