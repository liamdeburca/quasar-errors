from typing import Self, Literal, Callable
from numpy import empty, float64, hypot, pi
from scipy.stats import norm
from pydantic import validate_call
from pydantic.dataclasses import dataclass
from functools import cached_property

from quasar_models.line import GaussianModel, _VProfileCopy, evaluation
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import rv_continuous_frozen_

@dataclass(frozen=True)
class GaussianSample:
    """
    ** PYDANTIC DATACLASS **    
    """
    strength: float | FloatVector
    sigma_v: float | FloatVector
    v_off: float | FloatVector
    
    wave: float
    sigma_res: float
    n_sigmas: float
    
    name: str = ""
    pure_name: str = ""
    is_v_profile_copy: bool = False
    n_profiles: int = 1
    strength_scale: float = 1.0

    @classmethod
    @validate_call
    def fromGaussianModel(
        cls,
        model: GaussianModel | _VProfileCopy,
    ) -> Self:
        """
        Instantiates a GaussianSample from a GaussianModel or a velocity profile
        copy. 
        """
        if isinstance(model, GaussianModel):
            strength = model.strength.value
            sigma_v = model.sigma_v.value
            v_off = model.v_off.value
            wave = model.wave
            sigma_res = model.sigma_res
            n_sigmas = model.n_sigmas
            name = model.name
            pure_name = model.pure_name
            is_v_profile_copy = False
            n_profiles = 1
            strength_scale = 1.0

        elif isinstance(model, _VProfileCopy):
            n_profiles = model._n_profiles
            strength = empty(n_profiles, dtype=float64)
            sigma_v = empty(n_profiles, dtype=float64)
            v_off = empty(n_profiles, dtype=float64)
            for i in range(n_profiles):
                strength[i] = getattr(model, f"strength_{i}").value
                sigma_v[i] = getattr(model, f"sigma_v_{i}").value
                v_off[i] = getattr(model, f"v_off_{i}").value

            wave = model.wave
            sigma_res = model.sigma_res
            n_sigmas = model.n_sigmas
            name = pure_name = model.name
            is_v_profile_copy = True
            strength_scale = model.strength_scale.value

        else:
            msg = "Model must be either GaussianModel or velocity profile copy."
            raise TypeError(msg)
        

        return GaussianSample(
            strength,
            sigma_v,
            v_off,
            wave,
            sigma_res,
            n_sigmas,
            name=name,
            pure_name=pure_name,
            is_v_profile_copy=is_v_profile_copy,
            n_profiles=n_profiles,
            strength_scale=strength_scale,
        )
    
    def __call__(self, x: float | FloatVector) -> float | FloatVector:
        strength = self.strength
        sigma_v = self.sigma_v
        v_off = self.v_off
        if not self.is_v_profile_copy:
            strength = (strength,)
            sigma_v = (sigma_v,)
            v_off = (v_off,)

        return self.strength_scale * sum(
            evaluation.evaluate(
                x, 
                *args, 
                wave=self.wave, sigma_res=self.sigma_res,
            )
            for args in zip(strength, sigma_v, v_off)
        )

    def prime(self, x: float | FloatVector) -> float | FloatVector:
        strength = self.strength
        sigma_v = self.sigma_v
        v_off = self.v_off
        if not self.is_v_profile_copy:
            strength = (strength,)
            sigma_v = (sigma_v,)
            v_off = (v_off,)

        return self.strength_scale * sum(
            evaluation.prime(
                x, 
                *args, 
                wave=self.wave, sigma_res=self.sigma_res,
            )
            for args in zip(strength, sigma_v, v_off)
        )
    
    @property
    def model_type(self) -> Literal["em"]: 
        return "em"

    @property
    def sorting_key(self) -> tuple[float, float]: 
        return (4.0, self.wave)
    
    @cached_property
    def mu(self) -> float | FloatVector:
        return self.wave * (1 + self.v_off)
    
    @cached_property
    def sigma(self) -> float | FloatVector:
        return self.mu * hypot(self.sigma_v, self.sigma_res)
    
    @cached_property
    def peak(self) -> float:
        if self.is_v_profile_copy:
            msg = "Property 'peak' is not defined for velocity profile copies."
            raise NotImplementedError(msg)
        return self.strength / (self.sigma * (2 * pi)**0.5)
    
    @cached_property
    def dist(self) -> rv_continuous_frozen_ | list[rv_continuous_frozen_]:
        if self.is_v_profile_copy:
            return [norm(*args) for args in zip(self.mu, self.sigma)]
        else:
            return norm(self.mu, self.sigma)
    
    @cached_property
    def pdf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        kwargs = {}
        if self.is_v_profile_copy:
            kwargs['weight'] = self.strength / self.strength.sum()
            kwargs['dist'] = self.dist
        else:
            kwargs['weight'] = (1.0,)
            kwargs['dist'] = (self.dist,)

        def pdf(
            x: float | FloatVector, 
            *,
            kwargs: dict = kwargs,
        ) -> float | FloatVector:
            return sum(
                w * d.pdf(x)
                for w, d in zip(kwargs['weight'], kwargs['dist'])
            )

        return pdf
    
    @cached_property
    def cdf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        kwargs = {}
        if self.is_v_profile_copy:
            kwargs['weight'] = self.strength / self.strength.sum()
            kwargs['dist'] = self.dist
        else:
            kwargs['weight'] = (1.0,)
            kwargs['dist'] = (self.dist,)

        def cdf(
            x: float | FloatVector, 
            *, 
            kwargs: dict = kwargs,
        ) -> float | FloatVector:
            return sum(
                w * d.cdf(x)
                for w, d in zip(kwargs['weight'], kwargs['dist'])
            )

        return cdf
    
    @cached_property
    def isf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        if self.is_v_profile_copy and self.n_profiles > 1:
            msg = "Property 'isf' is not defined for multiple Gaussians."
            raise NotImplementedError(msg)

        if self.is_v_profile_copy:
            dist = self.dist[0]
        else:
            dist = self.dist

        def isf(q: float | FloatVector) -> float | FloatVector:
            return dist.isf(q)

        return isf
    
