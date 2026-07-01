from typing import Self, Literal, ClassVar
from numpy import array, zeros_like, float64

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from quasar_core.modelling.balmer import BalmerEvaluate, evaluate_exact

from quasar_models.balmer import (
    BalmerModel, 
    BalmerSeriesTemplate, BalmerContinuumTemplate,
)
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

from quasar_utils.decorators import validate_call

from ..utils.base_sample import BaseSample

@dataclass(frozen=True, kw_only=True)
class BalmerSample(BaseSample):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(BalmerEvaluate,),
    )

    flux: float
    fwhm: float
    ratio: float

    name: str = ""

    continuum_template: BalmerContinuumTemplate
    series_template: BalmerSeriesTemplate
    
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None
    evaluate_func: BalmerEvaluate = evaluate_exact

    model_type: ClassVar[Literal["ba"]] = "ba"

    @classmethod
    @validate_call
    def fromBalmerModel(cls, model: BalmerModel) -> Self:
        return BalmerSample(
            flux=model.flux.value,
            fwhm=model.fwhm.value,
            ratio=model.ratio.value,
            name=model.name,
            continuum_template=model.continuum_template,
            series_template=model.series_template,
            **model._interpolation_matrices,
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
        return self.evaluate_func(
            x,
            self.flux, self.fwhm, self.ratio,
            template_x=self.continuum_template.x,
            continuum_fwhm=self.continuum_template.fwhm,
            continuum_data=self.continuum_template.data / self.continuum_template.normalisation,
            series_fwhm=self.series_template.fwhm,
            series_data=self.series_template.data / self.series_template.normalisation,
            sigma_res=self.continuum_template.sigma_res,
            n_scales=self.continuum_template.n_scales,
            interpolation_matrix=self.interpolation_matrix,
            y=y,
        )
    
    @property
    def sorting_key(self) -> tuple[float, float]: 
        return (2.0, 0.0)