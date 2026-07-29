from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, Self

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function
from quasar_typing.numpy import FloatVector


class BaseSample(ABC):
    evaluate_func: Callable
    model_type: ClassVar[str]

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = f"Expected {cls.__name__}, got {type(value).__name__}"
            raise PydanticCustomError("validation_error", msg)
        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, source: type, handler: Callable):
        return no_info_plain_validator_function(cls._validate)

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
