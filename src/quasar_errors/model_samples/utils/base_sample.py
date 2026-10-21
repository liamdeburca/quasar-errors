from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import pydantic
from quasar_typing.numpy import FloatVector


class BaseSample(ABC, pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen=True, 
        extra="forbid", 
    )
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
