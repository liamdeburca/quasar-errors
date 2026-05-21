from typing import Self, ClassVar
from dataclasses import field
from pydantic.dataclasses import dataclass
from quasar_models.balmer import BalmerModel, BalmerContinuumTemplate, BalmerSeriesTemplate, evaluation
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_typing.scipy.sparse import csr_matrix_

from .utils import PseudoModel, PseudoParam

@dataclass
class PseudoBalmerModel(PseudoModel):
    flux: PseudoParam
    fwhm: PseudoParam
    ratio: PseudoParam

    name: str = field(kw_only=True)

    continuum_template: BalmerContinuumTemplate | None = field(kw_only=True)
    series_template: BalmerSeriesTemplate | None = field(kw_only=True)

    perform_interp_fitting: bool = field(kw_only=True)

    param_names: ClassVar[list[str]] = [
        'flux', 'fwhm', 'temp', 'tau', 'scale', 'ratio',
    ]
    n_params: ClassVar[int] = 6

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update({
            'name': self.name,
            'continuum_template': self.continuum_template,
            'series_template': self.series_template,
            'perform_interp_fitting': self.perform_interp_fitting,
        })
        return state
    
    def __setstate__(self, state: dict) -> None:
        self.name = state.pop('name')
        self.perform_interp_fitting = state.pop('perform_interp_fitting')
        self.continuum_template = state.pop('continuum_template')
        self.series_template = state.pop('series_template')
        super().__setstate__(state)

    @classmethod
    def from_model(cls, model: BalmerModel) -> Self:
        return PseudoBalmerModel(
            PseudoParam.from_parameter(model.flux),
            PseudoParam.from_parameter(model.fwhm),
            PseudoParam.from_parameter(model.ratio),
            name=model.name,
            continuum_template=model.continuum_template,
            series_template=model.series_template,
            perform_interp_fitting=model._perform_interp_fitting,
        )

    def evaluate(
        self, 
        x: float | FloatVector,
        flux: float,
        fwhm: float,
        ratio: float,
        *,
        continuum_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
        series_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    ) -> float | FloatVector:
        if self.perform_interp_fitting:
            return evaluation.evaluate_interp(
                x,
                flux, fwhm, ratio,
                continuum_template=self.continuum_template,
                series_template=self.series_template,
                continuum_interpolation_matrix=continuum_interpolation_matrix,
                series_interpolation_matrix=series_interpolation_matrix,
            )
        return evaluation.evaluate(
            x, 
            flux, fwhm, ratio,
            continuum_template=self.continuum_template,
            series_template=self.series_template,
            continuum_interpolation_matrix=continuum_interpolation_matrix,
            series_interpolation_matrix=series_interpolation_matrix,
        )
    
    def evaluate_sparse(
        self, 
        x: float | FloatVector,
        flux: float,
        fwhm: float,
        ratio: float,
        *,
        continuum_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
        series_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    ) -> tuple[BoolVector, FloatVector]:
        if self.perform_interp_fitting:
            return evaluation.evaluate_interp_sparse(
                x,
                flux, fwhm, ratio,
                continuum_template=self.continuum_template,
                series_template=self.series_template,
                continuum_interpolation_matrix=continuum_interpolation_matrix,
                series_interpolation_matrix=series_interpolation_matrix,
            )
        return evaluation.evaluate_sparse(
            x,
            flux, fwhm, ratio,
            continuum_template=self.continuum_template,
            series_template=self.series_template,
            continuum_interpolation_matrix=continuum_interpolation_matrix,
            series_interpolation_matrix=series_interpolation_matrix,
        )
    
    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        ratio: float,
        *,
        continuum_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
        series_interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
        y: FloatVector | None = None,
    ) -> FloatVector | None:
        if y is None:
            return self.evaluate(
                x, 
                flux, fwhm, ratio,
                continuum_interpolation_matrix=continuum_interpolation_matrix,
                series_interpolation_matrix=series_interpolation_matrix,
            )
        mask, _y = self.evaluate_sparse(
            x, 
            flux, fwhm, ratio,
            continuum_interpolation_matrix=continuum_interpolation_matrix,
            series_interpolation_matrix=series_interpolation_matrix,
        )
        y[mask] += _y[mask]