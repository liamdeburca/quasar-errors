from typing import Self, ClassVar
from pydantic.dataclasses import dataclass
from quasar_models.line import GaussianModel, evaluation
from quasar_typing.numpy import BoolVector, FloatVector

from .utils import PseudoModel, PseudoParam

@dataclass
class PseudoGaussianModel(PseudoModel):
    strength: PseudoParam
    sigma_v: PseudoParam
    v_off: PseudoParam

    name: str
    wave: float
    sigma_res: float
    n_sigmas: float

    param_names: ClassVar[list[str]] = ['strength', 'sigma_v', 'v_off']
    n_params: ClassVar[int] = 3

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state.update({
            'name': self.name,
            'wave': self.wave,
            'sigma_res': self.sigma_res,
            'n_sigmas': self.n_sigmas,
        })
        return state
    
    def __setstate__(self, state: dict) -> None:
        self.name = state.pop('name')
        self.wave = state.pop('wave')
        self.sigma_res = state.pop('sigma_res')
        self.n_sigmas = state.pop('n_sigmas')
        super().__setstate__(state)

    @classmethod
    def from_model(cls, model: GaussianModel) -> Self:
        return PseudoGaussianModel(
            PseudoParam.from_parameter(model.strength),
            PseudoParam.from_parameter(model.sigma_v),
            PseudoParam.from_parameter(model.v_off),
            model.name,
            model.wave,
            model.sigma_res,
            model.n_sigmas,
        )

    def evaluate(
        self, 
        x: float | FloatVector,
        strength: float,
        sigma_v: float,
        v_off: float,
    ) -> float | FloatVector:
        return evaluation.evaluate(
            x, 
            strength, sigma_v, v_off, 
            wave=self.wave, sigma_res=self.sigma_res,
        )
    
    def evaluate_sparse(
        self, 
        x: FloatVector,
        strength: float,
        sigma_v: float,
        v_off: float,
    ) -> tuple[BoolVector, FloatVector]:
        return evaluation.evaluate_sparse(
            x, 
            strength, sigma_v, v_off, 
            wave=self.wave, sigma_res=self.sigma_res, n_sigmas=self.n_sigmas,
        )
    
    def __call__(
        self,
        x: FloatVector,
        strength: float,
        sigma_v: float,
        v_off: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector:
        if y is None:
            return self.evaluate(x, strength, sigma_v, v_off)
        mask, _y = self.evaluate_sparse(x, strength, sigma_v, v_off)
        y[mask] += _y
        return y