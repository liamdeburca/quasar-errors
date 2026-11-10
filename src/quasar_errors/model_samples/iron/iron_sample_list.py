from collections.abc import Iterable
from typing import Literal, Self

from pydantic import validate_call
from quasar_models.iron import IronModel
from quasar_typing.astropy import ModelType

from ..utils.base_sample_list import BaseSampleList
from .iron_sample import IronSample


class IronSampleList(BaseSampleList[IronSample]):
    samples: list[IronSample]

    ### fromIronSamples

    @classmethod
    def _fromIronSamples(
        cls,
        samples: Iterable[IronSample],
    ) -> dict[Literal["samples"], list[IronSample]]:
        return {"samples": sorted(samples, key=lambda s: s.sorting_key)}

    @classmethod
    @validate_call
    def fromIronSamples(cls, samples: Iterable[IronSample]) -> Self:
        return cls(**cls._fromIronSamples(samples))

    ### fromIronSubmodels

    @classmethod
    def _fromIronSubmodels(cls, submodels: Iterable[IronModel]):
        return cls._fromIronSamples(
            IronSample.fromIronModel(model)
            for model in sorted(submodels, key=lambda m: m.sorting_key)
        )

    @classmethod
    @validate_call
    def fromIronSubmodels(cls, submodels: Iterable[IronModel]) -> Self:
        return cls(**cls._fromIronSubmodels(submodels))

    ### fromIronModels

    @classmethod
    def _fromIronModels(cls, model: ModelType):
        return cls._fromIronSubmodels(filter(
            lambda m: isinstance(m, IronModel),
            (model,) if model.n_submodels == 1 else model,
        ))

    @classmethod
    @validate_call
    def fromIronModels(cls, model: ModelType) -> Self:
        return cls(**cls._fromIronModels(model))
