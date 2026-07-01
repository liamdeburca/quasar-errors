from abc import ABC, abstractmethod
from typing import ClassVar, Callable
from quasar_typing.numpy import FloatVector

class BaseSample(ABC):
    evaluate_func: Callable
    model_type: ClassVar[str]

    @abstractmethod
    def __call__(
        self, 
        x: float | FloatVector, 
        y: FloatVector | None = None,
    ) -> float | FloatVector:
        pass

    @property
    @abstractmethod
    def sorting_key(self) -> tuple[float, float]:
        pass