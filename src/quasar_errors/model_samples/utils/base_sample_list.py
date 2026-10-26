from typing import Self, TypeVar

import numpy as np
import pydantic
from quasar_typing.numpy import FloatVector

from .base_sample import BaseSample

S = TypeVar("S", bound=BaseSample)


class BaseSampleList[S](pydantic.BaseModel):
    samples: list[S]

    @property
    def parameters(self) -> FloatVector:
        return np.concatenate([s.parameters for s in self.samples])


    @property
    def names(self) -> list[str]:
        return [s.name for s in self.samples]


    def sort(self, reverse: bool = False, inplace: bool = False) -> Self:
        if not inplace:
            return self.model_copy().sort(reverse=reverse, inplace=True)
        self.samples.sort(key=lambda s: s.sorting_key, reverse=reverse)
        return self


    def __call__(
        self, 
        x: float | FloatVector, 
        y: FloatVector | None = None
    ) -> float | FloatVector:
        if isinstance(x, (float, int)):
            assert y is None
            return self(np.array([x]), y=None)[0]

        for sample in self.samples:
            y = sample(x, y=y)
        return y


    def prime(
        self, 
        x: float | FloatVector, 
        y: FloatVector | None = None
    ) -> float | FloatVector:
        if isinstance(x, (float, int)):
            assert y is None
            return self.prime(np.array([x]), y=None)[0]

        for sample in self.samples:
            y = sample.prime(x, y=y)
        return y
