from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel
from quasar_typing.numpy import FloatVector


class BaseSample(ABC, BaseModel, frozen=True, extra="forbid"):
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
