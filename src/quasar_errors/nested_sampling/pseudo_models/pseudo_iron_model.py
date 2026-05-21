from typing import Self, ClassVar
from pydantic.dataclasses import dataclass
from numpy import place
from quasar_models.iron import IronModel, IronTemplate, evaluation
from quasar_typing.numpy import BoolVector, FloatVector
from quasar_typing.scipy.sparse import csr_matrix_

from .utils import PseudoModel, PseudoParam

@dataclass
class PseudoIronModel(PseudoModel):
    flux: PseudoParam
    fwhm: PseudoParam
    split: PseudoParam
    left: PseudoParam
    right: PseudoParam

    name: str
    template: IronTemplate
    scale: float
    sigma_res: float
    allow_interp_fitting: bool
    perform_interp_fitting: bool

    param_names: ClassVar[list[str]] = ['flux', 'fwhm', 'split', 'left', 'right']
    n_params: ClassVar[int] = 5

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update({
            'name': self.name,
            'template': self.template,
            'scale': self.scale,
            'sigma_res': self.sigma_res,
            'allow_interp_fitting': self.allow_interp_fitting,
            'perform_interp_fitting': self.perform_interp_fitting,
        })
        return state
    
    def __setstate__(self, state: dict) -> None:
        self.name = state.pop('name')
        self.template = state.pop('template')
        self.scale = state.pop('scale')
        self.sigma_res = state.pop('sigma_res')
        self.allow_interp_fitting = state.pop('allow_interp_fitting')
        self.perform_interp_fitting = state.pop('perform_interp_fitting')
        super().__setstate__(state)

    @classmethod
    def from_model(cls, model: IronModel) -> Self:
        return PseudoIronModel(
            PseudoParam.from_parameter(model.flux),
            PseudoParam.from_parameter(model.fwhm),
            PseudoParam.from_parameter(model.split),
            PseudoParam.from_parameter(model.left),
            PseudoParam.from_parameter(model.right),
            model.name,
            model.template,
            model.scale,
            model.sigma_res,
            model.allow_interp_fitting,
            model._perform_interp_fitting,
        )

    def evaluate(
        self, 
        x: float | FloatVector,
        flux: float,
        fwhm: float,
        split: float,
        left: float,
        right: float,
        *,
        interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    ) -> float | FloatVector:
        if self.perform_interp_fitting:
            return evaluation.evaluate_interp(
                x,
                flux, fwhm,
                template=self.template,
                interpolation_matrix=interpolation_matrix,
            )
        return evaluation.evaluate(
            x, 
            flux, fwhm, split, left, right, 
            template=self.template, scale=self.scale, sigma_res=self.sigma_res,
            interpolation_matrix=interpolation_matrix,
        )
    
    def evaluate_sparse(
        self, 
        x: FloatVector,
        flux: float,
        fwhm: float,
        split: float,
        left: float,
        right: float,
        *,
        interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
    ) -> tuple[BoolVector, FloatVector]:
        if self.perform_interp_fitting:
            return evaluation.evaluate_interp_sparse(
                x,
                flux, fwhm,
                template=self.template,
                interpolation_matrix=interpolation_matrix,
            )
        return evaluation.evaluate_sparse(
            x, 
            flux, fwhm, split, left, right, 
            template=self.template, scale=self.scale, sigma_res=self.sigma_res,
            interpolation_matrix=interpolation_matrix,
        )
    
    def __call__(
        self,
        x: FloatVector,
        flux: float,
        fwhm: float,
        split: float,
        left: float,
        right: float,
        *,
        interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
        y: FloatVector | None = None,
    ) -> FloatVector:
        if y is None:
            return self.evaluate(
                x, 
                flux, fwhm, split, left, right, 
                interpolation_matrix=interpolation_matrix,
            )
        place(
            y, 
            *self.evaluate_sparse(
                x, 
                flux, fwhm, split, left, right, 
                interpolation_matrix=interpolation_matrix,
            )
        )
        return y