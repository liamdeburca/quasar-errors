from typing import Self, Literal
from pydantic import validate_call
from pydantic.dataclasses import dataclass

from quasar_models.balmer import BalmerModel, BalmerSeriesTemplate, BalmerContinuumTemplate, evaluation
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

@dataclass(frozen=True)
class BalmerSample:
    """
    ** PYDANTIC DATACLASS **
    """
    flux: float
    fwhm: float
    temp: float
    tau: float
    scale: float
    ratio: float

    edge: float
    sigma_res: float
    waves: FloatVector
    weights: FloatVector
    boltz: float
    x_grid: FloatVector
    
    name: str = ""
    continuum_template: BalmerContinuumTemplate | None = None
    series_template: BalmerSeriesTemplate | None = None
    perform_interp_fitting: bool = False

    @classmethod
    @validate_call
    def fromBalmerModel(
        cls,
        model: BalmerModel,
    ) -> Self:
        return BalmerSample(
            model.flux.value,
            model.fwhm.value,
            model.temp.value,
            model.tau.value,
            model.scale.value,
            model.ratio.value,
            model.edge,
            model.sigma_res,
            model.waves,
            model.weights,
            model.boltz,
            model.x_grid,
            name=model.name,
            continuum_template=model.continuum_template,
            series_template=model.series_template,
            perform_interp_fitting=model._perform_interp_fitting,
        )
    
    def __call__(
        self,
        x: float | FloatVector,
        continuum_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
        series_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    ) -> float | FloatVector:
        if self.perform_interp_fitting:
            return evaluation.evaluate_interp(
                x,
                self.flux, self.fwhm, self.ratio,
                continuum_template=self.continuum_template,
                series_template=self.series_template, 
                continuum_interpolation_matrix=continuum_interpolation_matrix,
                series_interpolation_matrix=series_interpolation_matrix,
            )
        
        return evaluation.evaluate(
            x,
            self.flux, self.fwhm, self.ratio,
            continuum_template=self.continuum_template,
            series_template=self.series_template,
            continuum_interpolation_matrix=continuum_interpolation_matrix,
            series_interpolation_matrix=series_interpolation_matrix,
        )
    
    @property
    def model_type(self) -> Literal["ba"]: 
        return "ba"

    @property
    def sorting_key(self) -> tuple[float, float]: 
        return (2.0, 0.0)