from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Literal, Self

from numpy import array, float64, zeros_like
from quasar_models._core.modeling.powerlaw import PowerLawEvaluate, evaluate
from quasar_models.continuum import PowerLawModel
from quasar_typing.numpy import FloatVector
from quasar_utils.decorators import validate_call

from ..utils.base_sample import BaseSample


@dataclass(frozen=True, kw_only=True)
class PowerLawSample(BaseSample):
    flux: float
    alpha: float
    x0: float

    name: str

    evaluate_func: PowerLawEvaluate = evaluate

    model_type: ClassVar[Literal["pl"]] = "pl"

    @classmethod
    @validate_call
    def fromPowerLawModel(
        cls,
        model: PowerLawModel,
    ) -> Self:
        return PowerLawSample(
            flux=model.flux.value,
            alpha=model.alpha.value,
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
            _x = array([x], dtype=float64)
            return self(_x)[0]

        if y is None:
            y = zeros_like(x, dtype=float64)
        return self.evaluate_func(x, self.flux, self.alpha, x0=self.x0, y=y)

    @cached_property
    def sorting_key(self) -> tuple[float, float]:
        return (0.0, self.x0)
