from typing import Iterator, TypeVar
from numpy import array, zeros_like, float64

from quasar_typing.numpy import FloatVector
from .base_sample import BaseSample

S = TypeVar('S', bound=BaseSample)

class BaseSampleList[S]:
    def __iter__(self) -> Iterator[S]:
        return iter(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __bool__(self) -> bool:
        return len(self.samples) > 0

    def __getitem__(self, index: int | slice) -> S | list[S]:
        return self.samples[index]
    
    def __call__(
        self, 
        x: float | FloatVector,
        y: FloatVector | None = None
    ) -> float | FloatVector:
        if isinstance(x, (float, int)):
            assert y is None
            return sum(sample(x) for sample in self)

        if y is None:
            y = zeros_like(x, dtype=float64)

        for sample in self:
            sample(x, y=y)
        return y
    
    def prime(
        self, 
        x: float | FloatVector,
        y: FloatVector | None = None
    ) -> float | FloatVector:
        if isinstance(x, (float, int)):
            assert y is None
            _x = array([x], dtype=float64)
            return self.prime(_x)[0]
        
        if y is None:
            y = zeros_like(x, dtype=float64)

        for sample in self:
            sample.prime(x, y=y)
        return y