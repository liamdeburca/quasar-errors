from functools import cached_property
from typing import ClassVar, Literal, Self

import numpy as np
import pydantic
from quasar_models._core.modeling.powerlaw import PowerLawEvaluate, evaluate
from quasar_models.continuum import PowerLawModel
from quasar_typing.numpy import FloatVector

from ..utils.base_sample import BaseSample


class PowerLawSample(BaseSample):
    model_config = pydantic.ConfigDict(
        frozen=True, 
        extra="forbid", 
        arbitrary_types_allowed=True,
    )

    flux: float
    alpha: float
    x0: float
    name: str
    evaluate_func: PowerLawEvaluate = evaluate

    model_type: ClassVar[Literal["pl"]] = "pl"

    @cached_property
    def parameters(self) -> FloatVector:
        return np.array([self.flux, self.alpha], dtype=np.float64)

    @classmethod
    def _validate_is_prepared(cls, model: PowerLawModel) -> None:
        if "evaluate_func" not in model.meta:
            raise ValueError(
                "PowerLaw model instance was not properly prepared prior to "
                "error estimation! This could be hugely inefficient."
            )

    @classmethod
    @pydantic.validate_call
    def fromPowerLawModel(
        cls,
        model: PowerLawModel,
    ) -> Self:
        cls._validate_is_prepared(model)
        return PowerLawSample(
            flux=model.flux.value,
            alpha=model.alpha.value,
            x0=model.x0,
            name=model.name,
            evaluate_func=model.evaluate_func,
        )

    @classmethod
    @pydantic.validate_call
    def fromParamVector(
        cls,
        param_vector: FloatVector,
        model: PowerLawModel,
    ) -> Self:
        cls._validate_is_prepared(model)
        return PowerLawSample(
            flux=param_vector[0],
            alpha=param_vector[1],
            x0=model.x0,
            name=model.name,
            evaluate_func=model.evaluate_func,
        )

    def __call__(
        self,
        x: float | FloatVector,
        y: FloatVector | None = None,
    ) -> float | FloatVector:
        if isinstance(x, (int, float)):
            assert y is None
            _x = np.array([x], dtype=np.float64)
            return self(_x)[0]

        if y is None:
            y = np.zeros_like(x, dtype=np.float64)
        return self.evaluate_func(x, self.flux, self.alpha, x0=self.x0, y=y)

    @cached_property
    def sorting_key(self) -> tuple[float, float]:
        return (0.0, 0.0)
