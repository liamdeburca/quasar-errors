from typing import Self, ClassVar, Iterable
from numpy import zeros_like, float64, bool_
from pydantic.dataclasses import dataclass
from quasar_models.line import evaluation, vprofilecopies
from quasar_typing.numpy import BoolVector, FloatVector

from .utils import PseudoModel, PseudoParam

class _PseudoVProfileCopy(PseudoModel):
    name: str
    wave: float
    sigma_res: float
    n_sigmas: float
    
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

    def evaluate(
        self,
        x: float | FloatVector,
        strength_scale: float,
        strengths: Iterable[float],
        sigma_vs: Iterable[float],
        v_offs: Iterable[float],
    ) -> float | FloatVector:
        contribs = (
            evaluation.evaluate(x, *a, wave=self.wave, sigma_res=self.sigma_res)
            for a in zip(strengths, sigma_vs, v_offs)
        )        
        return strength_scale * sum(contribs)
    
    def evaluate_sparse(
        self,
        x: FloatVector,
        strength_scale: float,
        strengths: Iterable[float],
        sigma_vs: Iterable[float],
        v_offs: Iterable[float],
    ) -> tuple[BoolVector, FloatVector]:
        y = zeros_like(x, dtype=float64)
        mask = zeros_like(x, dtype=bool_)
        for strength, sigma_v, v_off in zip(strengths, sigma_vs, v_offs):
            _mask, _y = evaluation.evaluate_sparse(
                    x,
                    strength_scale * strength, sigma_v, v_off,
                    wave=self.wave,
                    sigma_res=self.sigma_res,
                    n_sigmas=self.n_sigmas,
                )
            y[_mask] += _y
            mask |= _mask

        return mask, y
    
    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strengths: Iterable[float],
        sigma_vs: Iterable[float],
        v_offs: Iterable[float],
        *,
        y: FloatVector | None = None,
    ) -> FloatVector | None:
        if y is None:
            return self.evaluate(x, strength_scale, strengths, sigma_vs, v_offs)
        mask, _y = self.evaluate_sparse(
            x, 
            strength_scale, strengths, sigma_vs, v_offs,
        )
        y[mask] += _y[mask]
    
@dataclass
class PseudoVProfileCopy1G(_PseudoVProfileCopy):
    strength_scale: PseudoParam
    
    strength_1: PseudoParam
    sigma_v_1: PseudoParam
    v_off_1: PseudoParam

    name: str
    wave: float
    sigma_res: float
    n_sigmas: float

    n_profiles: ClassVar[int] = 1
    param_names: ClassVar[list[str]] = [
        'strength_scale', 
        'strength_1', 'sigma_v_1', 'v_off_1',
    ]
    n_params: ClassVar[int] = 4

    @classmethod
    def from_model(cls, model: vprofilecopies.VProfileCopy1G) -> Self:
        return PseudoVProfileCopy1G(
            PseudoParam.from_parameter(model.strength_scale),
            PseudoParam.from_parameter(model.strength_1),
            PseudoParam.from_parameter(model.sigma_v_1),
            PseudoParam.from_parameter(model.v_off_1),
            model.name,
            model.wave,
            model.sigma_res,
            model.n_sigmas,
        )
    
    def evaluate(
        self,
        x: float | FloatVector,
        strength_scale: float,
        strength_1: float,
        sigma_v_1: float,
        v_off_1: float,
    ) -> float | FloatVector:
        return super().evaluate(
            x, strength_scale, 
            (strength_1,), (sigma_v_1,), (v_off_1,),
        )
    
    def evaluate_sparse(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float,
        sigma_v_1: float,
        v_off_1: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector:
        return super().evaluate_sparse(
            x, strength_scale, 
            (strength_1,), (sigma_v_1,), (v_off_1,),
            y=y,
        )
    
    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float,
        sigma_v_1: float,
        v_off_1: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector | None:
        return super().__call__(
            x, strength_scale, 
            (strength_1,), (sigma_v_1,), (v_off_1,),
            y=y,
    )
    
@dataclass
class PseudoVProfileCopy2G(_PseudoVProfileCopy):
    strength_scale: PseudoParam
    
    strength_1: PseudoParam
    sigma_v_1: PseudoParam
    v_off_1: PseudoParam

    strength_2: PseudoParam
    sigma_v_2: PseudoParam
    v_off_2: PseudoParam

    name: str
    wave: float
    sigma_res: float
    n_sigmas: float

    n_profiles: ClassVar[int] = 2
    param_names: ClassVar[list[str]] = [
        'strength_scale', 
        'strength_1', 'sigma_v_1', 'v_off_1',
        'strength_2', 'sigma_v_2', 'v_off_2',
    ]
    n_params: ClassVar[int] = 7

    @classmethod
    def from_model(cls, model: vprofilecopies.VProfileCopy2G) -> Self:
        return PseudoVProfileCopy2G(
            PseudoParam.from_parameter(model.strength_scale),
            PseudoParam.from_parameter(model.strength_1),
            PseudoParam.from_parameter(model.sigma_v_1),
            PseudoParam.from_parameter(model.v_off_1),
            PseudoParam.from_parameter(model.strength_2),
            PseudoParam.from_parameter(model.sigma_v_2),
            PseudoParam.from_parameter(model.v_off_2),
            model.name,
            model.wave,
            model.sigma_res,
            model.n_sigmas,
        )
    
    def evaluate(
        self,
        x: float | FloatVector,
        strength_scale: float, 
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
    ) -> float | FloatVector:
        return super().evaluate(
            x, strength_scale, 
            (strength_1, strength_2), 
            (sigma_v_1, sigma_v_2), 
            (v_off_1, v_off_2),
        )
    
    def evaluate_sparse(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector:
        return super().evaluate_sparse(
            x, strength_scale, 
            (strength_1, strength_2), 
            (sigma_v_1, sigma_v_2), 
            (v_off_1, v_off_2),
            y=y,
        )
    
    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector | None:
        return super().__call__(
            x, strength_scale, 
            (strength_1, strength_2), 
            (sigma_v_1, sigma_v_2), 
            (v_off_1, v_off_2),
            y=y,
    )
    
@dataclass
class PseudoVProfileCopy3G(_PseudoVProfileCopy):
    strength_scale: PseudoParam
    
    strength_1: PseudoParam
    sigma_v_1: PseudoParam
    v_off_1: PseudoParam

    strength_2: PseudoParam
    sigma_v_2: PseudoParam
    v_off_2: PseudoParam

    strength_3: PseudoParam
    sigma_v_3: PseudoParam
    v_off_3: PseudoParam

    name: str
    wave: float
    sigma_res: float
    n_sigmas: float

    n_profiles: ClassVar[int] = 3
    param_names: ClassVar[list[str]] = [
        'strength_scale', 
        'strength_1', 'sigma_v_1', 'v_off_1',
        'strength_2', 'sigma_v_2', 'v_off_2',
        'strength_3', 'sigma_v_3', 'v_off_3',
    ]
    n_params: ClassVar[int] = 10

    @classmethod
    def from_model(cls, model: vprofilecopies.VProfileCopy3G) -> Self:
        return PseudoVProfileCopy3G(
            PseudoParam.from_parameter(model.strength_scale),
            PseudoParam.from_parameter(model.strength_1),
            PseudoParam.from_parameter(model.sigma_v_1),
            PseudoParam.from_parameter(model.v_off_1),
            PseudoParam.from_parameter(model.strength_2),
            PseudoParam.from_parameter(model.sigma_v_2),
            PseudoParam.from_parameter(model.v_off_2),
            PseudoParam.from_parameter(model.strength_3),
            PseudoParam.from_parameter(model.sigma_v_3),
            PseudoParam.from_parameter(model.v_off_3),
            model.name,
            model.wave,
            model.sigma_res,
            model.n_sigmas,
        )
    
    def evaluate(
        self,
        x: float | FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        strength_3: float, sigma_v_3: float, v_off_3: float,
    ) -> float | FloatVector:
        return super().evaluate(
            x, strength_scale,
            (strength_1, strength_2, strength_3),
            (sigma_v_1, sigma_v_2, sigma_v_3),
            (v_off_1, v_off_2, v_off_3),
        )
    
    def evaluate_sparse(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        strength_3: float, sigma_v_3: float, v_off_3: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector:
        return super().evaluate_sparse(
            x, strength_scale,
            (strength_1, strength_2, strength_3),
            (sigma_v_1, sigma_v_2, sigma_v_3),
            (v_off_1, v_off_2, v_off_3),
            y=y,
        )
    
    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        strength_3: float, sigma_v_3: float, v_off_3: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector | None:
        return super().__call__(
            x, strength_scale, 
            (strength_1, strength_2, strength_3), 
            (sigma_v_1, sigma_v_2, sigma_v_3), 
            (v_off_1, v_off_2, v_off_3),
            y=y,
        )

@dataclass
class PseudoVProfileCopy4G(_PseudoVProfileCopy):
    strength_scale: PseudoParam
    
    strength_1: PseudoParam
    sigma_v_1: PseudoParam
    v_off_1: PseudoParam

    strength_2: PseudoParam
    sigma_v_2: PseudoParam
    v_off_2: PseudoParam

    strength_3: PseudoParam
    sigma_v_3: PseudoParam
    v_off_3: PseudoParam

    strength_4: PseudoParam
    sigma_v_4: PseudoParam
    v_off_4: PseudoParam

    name: str
    wave: float
    sigma_res: float
    n_sigmas: float

    n_profiles: ClassVar[int] = 4
    param_names: ClassVar[list[str]] = [
        'strength_scale', 
        'strength_1', 'sigma_v_1', 'v_off_1',
        'strength_2', 'sigma_v_2', 'v_off_2',
        'strength_3', 'sigma_v_3', 'v_off_3',
        'strength_4', 'sigma_v_4', 'v_off_4',
    ]
    n_params: ClassVar[int] = 13

    @classmethod
    def from_model(cls, model: vprofilecopies.VProfileCopy4G) -> Self:
        return PseudoVProfileCopy4G(
            PseudoParam.from_parameter(model.strength_scale),
            PseudoParam.from_parameter(model.strength_1),
            PseudoParam.from_parameter(model.sigma_v_1),
            PseudoParam.from_parameter(model.v_off_1),
            PseudoParam.from_parameter(model.strength_2),
            PseudoParam.from_parameter(model.sigma_v_2),
            PseudoParam.from_parameter(model.v_off_2),
            PseudoParam.from_parameter(model.strength_3),
            PseudoParam.from_parameter(model.sigma_v_3),
            PseudoParam.from_parameter(model.v_off_3),
            PseudoParam.from_parameter(model.strength_4),
            PseudoParam.from_parameter(model.sigma_v_4),
            PseudoParam.from_parameter(model.v_off_4),
            model.name,
            model.wave,
            model.sigma_res,
            model.n_sigmas,
        )
    
    def evaluate(
        self,
        x: float | FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        strength_3: float, sigma_v_3: float, v_off_3: float,
        strength_4: float, sigma_v_4: float, v_off_4: float,
    ) -> float | FloatVector:
        return super().evaluate(
            x, strength_scale,
            (strength_1, strength_2, strength_3, strength_4),
            (sigma_v_1, sigma_v_2, sigma_v_3, sigma_v_4),
            (v_off_1, v_off_2, v_off_3, v_off_4),
        )
    
    def evaluate_sparse(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        strength_3: float, sigma_v_3: float, v_off_3: float,
        strength_4: float, sigma_v_4: float, v_off_4: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector:
        return super().evaluate_sparse(
            x, strength_scale,
            (strength_1, strength_2, strength_3, strength_4),
            (sigma_v_1, sigma_v_2, sigma_v_3, sigma_v_4),
            (v_off_1, v_off_2, v_off_3, v_off_4),
            y=y,
        )
    
    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        strength_3: float, sigma_v_3: float, v_off_3: float,
        strength_4: float, sigma_v_4: float, v_off_4: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector | None:
        return super().__call__(
            x, strength_scale, 
            (strength_1, strength_2, strength_3, strength_4), 
            (sigma_v_1, sigma_v_2, sigma_v_3, sigma_v_4), 
            (v_off_1, v_off_2, v_off_3, v_off_4),
            y=y,
        )

@dataclass
class PseudoVProfileCopy5G(_PseudoVProfileCopy):
    strength_scale: PseudoParam
    
    strength_1: PseudoParam
    sigma_v_1: PseudoParam
    v_off_1: PseudoParam

    strength_2: PseudoParam
    sigma_v_2: PseudoParam
    v_off_2: PseudoParam

    strength_3: PseudoParam
    sigma_v_3: PseudoParam
    v_off_3: PseudoParam

    strength_4: PseudoParam
    sigma_v_4: PseudoParam
    v_off_4: PseudoParam

    strength_5: PseudoParam
    sigma_v_5: PseudoParam
    v_off_5: PseudoParam

    name: str
    wave: float
    sigma_res: float
    n_sigmas: float

    n_profiles: ClassVar[int] = 5
    param_names: ClassVar[list[str]] = [
        'strength_scale', 
        'strength_1', 'sigma_v_1', 'v_off_1',
        'strength_2', 'sigma_v_2', 'v_off_2',
        'strength_3', 'sigma_v_3', 'v_off_3',
        'strength_4', 'sigma_v_4', 'v_off_4',
        'strength_5', 'sigma_v_5', 'v_off_5',
    ]
    n_params: ClassVar[int] = 16

    @classmethod
    def from_model(cls, model: vprofilecopies.VProfileCopy5G) -> Self:
        return PseudoVProfileCopy5G(
            PseudoParam.from_parameter(model.strength_scale),
            PseudoParam.from_parameter(model.strength_1),
            PseudoParam.from_parameter(model.sigma_v_1),
            PseudoParam.from_parameter(model.v_off_1),
            PseudoParam.from_parameter(model.strength_2),
            PseudoParam.from_parameter(model.sigma_v_2),
            PseudoParam.from_parameter(model.v_off_2),
            PseudoParam.from_parameter(model.strength_3),
            PseudoParam.from_parameter(model.sigma_v_3),
            PseudoParam.from_parameter(model.v_off_3),
            PseudoParam.from_parameter(model.strength_4),
            PseudoParam.from_parameter(model.sigma_v_4),
            PseudoParam.from_parameter(model.v_off_4),
            PseudoParam.from_parameter(model.strength_5),
            PseudoParam.from_parameter(model.sigma_v_5),
            PseudoParam.from_parameter(model.v_off_5),
            model.name,
            model.wave,
            model.sigma_res,
            model.n_sigmas,
        )
    
    def evaluate(
        self,
        x: float | FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        strength_3: float, sigma_v_3: float, v_off_3: float,
        strength_4: float, sigma_v_4: float, v_off_4: float,
        strength_5: float, sigma_v_5: float, v_off_5: float,
    ) -> float | FloatVector:
        return super().evaluate(
            x, strength_scale,
            (strength_1, strength_2, strength_3, strength_4, strength_5),
            (sigma_v_1, sigma_v_2, sigma_v_3, sigma_v_4, sigma_v_5),
            (v_off_1, v_off_2, v_off_3, v_off_4, v_off_5),
        )
    
    def evaluate_sparse(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        strength_3: float, sigma_v_3: float, v_off_3: float,
        strength_4: float, sigma_v_4: float, v_off_4: float,
        strength_5: float, sigma_v_5: float, v_off_5: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector:
        return super().evaluate_sparse(
            x, strength_scale,
            (strength_1, strength_2, strength_3, strength_4, strength_5),
            (sigma_v_1, sigma_v_2, sigma_v_3, sigma_v_4, sigma_v_5),
            (v_off_1, v_off_2, v_off_3, v_off_4, v_off_5),
            y=y,
        )
    
    def __call__(
        self,
        x: FloatVector,
        strength_scale: float,
        strength_1: float, sigma_v_1: float, v_off_1: float, 
        strength_2: float, sigma_v_2: float, v_off_2: float,
        strength_3: float, sigma_v_3: float, v_off_3: float,
        strength_4: float, sigma_v_4: float, v_off_4: float,
        strength_5: float, sigma_v_5: float, v_off_5: float,
        *,
        y: FloatVector | None = None,
    ) -> FloatVector | None:
        return super().__call__(
            x, strength_scale, 
            (strength_1, strength_2, strength_3, strength_4, strength_5), 
            (sigma_v_1, sigma_v_2, sigma_v_3, sigma_v_4, sigma_v_5), 
            (v_off_1, v_off_2, v_off_3, v_off_4, v_off_5),
            y=y,
        )