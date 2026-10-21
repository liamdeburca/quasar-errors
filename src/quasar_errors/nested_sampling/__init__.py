__all__ = [
    "BaseNestedSampler",
    "UniformNestedSampler",
]

from typing import ClassVar, Literal

import pydantic

from .base_nested_sampler import BaseNestedSampler


@pydantic.dataclasses.dataclass
class UniformNestedSampler(BaseNestedSampler):
    prior: ClassVar[Literal["uniform"]] = "uniform"
    loss: ClassVar[Literal["chi2"]] = "chi2"
