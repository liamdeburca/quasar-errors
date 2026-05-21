__all__ = ['UniformNestedSampler']

from typing import ClassVar, Literal
from pydantic.dataclasses import dataclass
from .base_nested_sampler import BaseNestedSampler

@dataclass
class UniformNestedSampler(BaseNestedSampler):
    prior: ClassVar[Literal['uniform']] = 'uniform'
    loss: ClassVar[Literal['chi2']] = 'chi2'