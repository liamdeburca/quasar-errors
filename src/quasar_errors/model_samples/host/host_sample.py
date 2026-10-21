from functools import cached_property
from typing import ClassVar, Literal, Self

import numpy as np
import pydantic
from quasar_models._core.modeling.host import (
    HostGalaxyEvaluate,
    evaluate_exact,
)
from quasar_models._core.modeling.template.cytemplate import CyTemplate
from quasar_models.host import (
    HostGalaxyModel,
    HostGalaxyTemplate,
)
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import CSRMatrix

from ..utils.base_sample import BaseSample


class HostGalaxySample(BaseSample):
    model_config = pydantic.ConfigDict(
        frozen=True, 
        extra="forbid", 
        arbitrary_types_allowed=True,
    )
    
    flux: float
    fwhm: float

    name: str
    template: HostGalaxyTemplate
    cytemplate: CyTemplate
    n_scales: float
    interpolation_matrix: tuple[CSRMatrix, FloatVector] | None
    
    evaluate_func: HostGalaxyEvaluate = evaluate_exact

    model_type: ClassVar[Literal["hg"]] = "hg"

    @cached_property
    def parameters(self) -> FloatVector:
        return np.array([self.flux, self.fwhm], dtype=np.float64)

    @classmethod
    def _validate_is_prepared(cls, model: HostGalaxyModel) -> None:
        if model.cytemplates["cytemplate"] is None:
            raise ValueError(
                "Host galaxy model instance was not properly prepared prior to "
                "error estimation! This could be hugely inefficient."
            )

    @classmethod
    @pydantic.validate_call
    def fromHostGalaxyModel(cls, model: HostGalaxyModel) -> Self:
        cls._validate_is_prepared(model)
        return HostGalaxySample(
            flux=model.flux.value,
            fwhm=model.fwhm.value,
            name=model.name,
            template=model.template,
            n_scales=model.n_scales,
            interpolation_matrix=model.interpolation_matrix,
            evaluate_func=model.evaluate_func,
            **model.cytemplates,
        )

    @classmethod
    @pydantic.validate_call
    def fromParamVector(
        cls,
        param_vector: FloatVector,
        model: HostGalaxyModel,
    ) -> Self:
        cls._validate_is_prepared(model)
        return HostGalaxySample(
            flux=param_vector[0],
            fwhm=param_vector[1],
            name=model.name,
            template=model.template,
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
            self.flux, self.fwhm,
            template=self.cytemplate,
            n_scales=self.n_scales,
            interpolation_matrix=self.interpolation_matrix,
            y=y,
        )

    @property
    def sorting_key(self) -> tuple[float, float]:
        return (3.0, 0.0)
