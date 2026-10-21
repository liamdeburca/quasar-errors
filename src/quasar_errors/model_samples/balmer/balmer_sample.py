from functools import cached_property
from typing import ClassVar, Literal, Self

import numpy as np
import pydantic
from quasar_models._core.modeling.balmer import BalmerEvaluate, evaluate_exact
from quasar_models._core.modeling.template.cytemplate import CyTemplate
from quasar_models.balmer import (
    BalmerContinuumTemplate,
    BalmerModel,
    BalmerSeriesTemplate,
)
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import CSRMatrix

from ..utils.base_sample import BaseSample


class BalmerSample(BaseSample):
    model_config = pydantic.ConfigDict(
        frozen=True, 
        extra="forbid", 
        arbitrary_types_allowed=True,
    )

    flux: float
    fwhm: float
    ratio: float

    name: str
    continuum_template: BalmerContinuumTemplate
    series_template: BalmerSeriesTemplate
    continuum_cytemplate: CyTemplate
    series_cytemplate: CyTemplate
    n_scales: float
    interpolation_matrix: tuple[CSRMatrix, FloatVector] | None

    evaluate_func: BalmerEvaluate = evaluate_exact

    model_type: ClassVar[Literal["ba"]] = "ba"

    @cached_property
    def parameters(self) -> FloatVector:
        return np.array([self.flux, self.fwhm, self.ratio], dtype=np.float64)

    @classmethod
    def _validate_is_prepared(cls, model: BalmerModel) -> None:
        continuum_cytemplate = model.cytemplates["continuum_cytemplate"]
        series_cytemplate = model.cytemplates["series_cytemplate"]
        if continuum_cytemplate is None or series_cytemplate is None:
            raise ValueError(
                "Balmer model instance was not properly prepared prior to "
                "error estimation! This could be hugely inefficient."
            )

    @classmethod
    @pydantic.validate_call
    def fromBalmerModel(cls, model: BalmerModel) -> Self:
        cls._validate_is_prepared(model)
        return BalmerSample(
            flux=model.flux.value,
            fwhm=model.fwhm.value,
            ratio=model.ratio.value,
            name=model.name,
            continuum_template=model.continuum_template,
            series_template=model.series_template,
            n_scales=model.n_scales,
            interpolation_matrix=model.interpolation_matrix,
            evaluate_func=model.evaluate_func,
            **model.cytemplates
        )

    @classmethod
    @pydantic.validate_call
    def fromParamVector(
        cls,
        param_vector: FloatVector,
        model: BalmerModel,
    ) -> Self:
        cls._validate_is_prepared(model)
        return BalmerSample(
            flux=param_vector[0],
            fwhm=param_vector[1],
            ratio=param_vector[2],
            name=model.name,
            continuum_template=model.continuum_template,
            series_template=model.series_template,
            n_scales=model.n_scales,
            interpolation_matrix=model.interpolation_matrix,
            evaluate_func=model.evaluate_func,
            **model.cytemplates,
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

        return self.evaluate_func(
            x,
            self.flux, self.fwhm, self.ratio,
            continuum_template=self.continuum_cytemplate,
            series_template=self.series_cytemplate,
            n_scales=self.n_scales,
            interpolation_matrix=self.interpolation_matrix,
            y=y,
        )

    @property
    def sorting_key(self) -> tuple[float, float]:
        return (2.0, 0.0)
