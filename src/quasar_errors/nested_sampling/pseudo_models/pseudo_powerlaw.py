from typing import Self, ClassVar
from pydantic.dataclasses import dataclass
from quasar_models.continuum import PowerLawModel, evaluation
from quasar_typing.numpy import BoolVector, FloatVector

from .utils import PseudoModel, PseudoParam

@dataclass
class PseudoPowerLawModel(PseudoModel):
    flux: PseudoParam
    alpha: PseudoParam

    name: str
    x0: float
    
    param_names: ClassVar[list[str]] = ['flux', 'alpha']
    n_params: ClassVar[int] = 2

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update({
            'name': self.name,
            'x0': self.x0,
        })
        return state
    
    def __setstate__(self, state: dict) -> None:
        self.name = state.pop('name')
        self.x0 = state.pop('x0')
        super().__setstate__(state)

    @classmethod
    def from_model(cls, model: PowerLawModel) -> Self:
        return PseudoPowerLawModel(
            PseudoParam.from_parameter(model.flux),
            PseudoParam.from_parameter(model.alpha),
            model.name,
            model.x0,
        )
    
    def evaluate(
        self, 
        x: float | FloatVector,
        flux: float,
        alpha: float,
    ) -> float | FloatVector:
        return evaluation.evaluate(x, flux, alpha, self.x0)
    
    def evaluate_sparse(
        self, 
        x: FloatVector,
        flux: float,
        alpha: float,
    ) -> tuple[BoolVector, FloatVector]:
        return evaluation.evaluate_sparse(x, flux, alpha, self.x0)
    
    def __call__(
        self,
        x: FloatVector,
        flux: float,
        alpha: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector | None:
        if y is None:
            return self.evaluate(x, flux, alpha)
        mask, _y = self.evaluate_sparse(x, flux, alpha)
        y[mask] += _y[mask]
