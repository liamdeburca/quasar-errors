from dataclasses import dataclass
from typing import ClassVar, Literal, Self

from numpy import array, float64, zeros_like
from quasar_models._core.modeling.balmer import BalmerEvaluate, evaluate_exact
from quasar_models._core.modeling.template.cytemplate import CyTemplate
from quasar_models.balmer import (
    BalmerContinuumTemplate,
    BalmerModel,
    BalmerSeriesTemplate,
)
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_utils.decorators import validate_call

from ..utils.base_sample import BaseSample


@dataclass(frozen=True, kw_only=True)
class BalmerSample(BaseSample):
    flux: float
    fwhm: float
    ratio: float

    name: str
    continuum_template: BalmerContinuumTemplate
    series_template: BalmerSeriesTemplate
    continuum_cytemplate: CyTemplate | None
    series_cytemplate: CyTemplate | None
    n_scales: float
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None

    evaluate_func: BalmerEvaluate = evaluate_exact

    model_type: ClassVar[Literal["ba"]] = "ba"

    @classmethod
    @validate_call
    def fromBalmerModel(cls, model: BalmerModel) -> Self:
        cytemplates = model.cytemplates        
        return BalmerSample(
            flux=model.flux.value,
            fwhm=model.fwhm.value,
            ratio=model.ratio.value,
            name=model.name,
            continuum_template=model.continuum_template,
            series_template=model.series_template,
            continuum_cytemplate=cytemplates['continuum_cytemplate'],
            series_cytemplate=cytemplates['series_cytemplate'],
            interpolation_matrix=model.interpolation_matrix,
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
            continuum_template=self.continuum_template,
            series_template=self.series_template,
            continuum_cytemplate=self.continuum_cytemplate,
            series_cytemplate=self.series_cytemplate,
            n_scales=self.n_scales,
            interpolation_matrix=self.interpolation_matrix,
            y=y,
        )

    @property
    def sorting_key(self) -> tuple[float, float]:
        return (2.0, 0.0)
