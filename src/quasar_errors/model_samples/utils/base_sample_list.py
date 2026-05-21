from typing import Iterator, TypeVar
from quasar_typing.numpy import FloatVector

T = TypeVar('T', bound='BaseSampleList')

class BaseSampleList[T]:
    def __iter__(self) -> Iterator[T]:
        return iter(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __bool__(self) -> bool:
        return len(self.samples) > 0

    def __call__(self, x: float | FloatVector) -> float | FloatVector:
        return sum(sample(x) for sample in self)
    
    def prime(self, x: float | FloatVector) -> float | FloatVector:
        return sum(sample.prime(x) for sample in self)

    def __getitem__(self, index: int | slice) -> T | list[T]:
        return self.samples[index]
        