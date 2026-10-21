import math
from collections.abc import Callable
from dataclasses import field
from functools import cached_property
from typing import Any, ClassVar, Literal, Self

import numpy as np
import pydantic
from astropy.constants import c
from quasar_models import EmModel, GaussianModel
from quasar_models._core.modeling.gaussian import (
    GaussianEvaluateV,
    choose_evaluate_func,
    choose_prime_func,
    evaluate_v,
    prime_v,
)
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import RVContinuousFrozen
from scipy.stats import norm

from ..utils.base_sample import BaseSample

GAUSS_AMP = 1 / (2 * math.pi) ** 0.5
SIGMA_TO_FWHM = 2 * (2 * math.log(2)) ** 0.5
FWHM_TO_SIGMA = 1 / SIGMA_TO_FWHM
C_KMS: float = c.to("km/s").value


class GaussianSample(BaseSample):
    model_config = pydantic.ConfigDict(
        frozen=True, 
        extra="forbid", 
        arbitrary_types_allowed=True,
    )
    strength: float | FloatVector
    fwhm_v: float | FloatVector
    v_off: float | FloatVector

    name: str
    pure_name: str
    wave: float
    n_sigmas: float
    is_v_profile_copy: bool
    n_profiles: int
    strength_scale: float
    linetype: Literal["n", "b"]

    dx: float | None
    sigma_res: float | None

    kwargs: dict[str, Any] = field(default_factory=dict)
    evaluate_func: GaussianEvaluateV = evaluate_v
    prime_func: GaussianEvaluateV = prime_v

    model_type: ClassVar[Literal["em"]] = "em"

    @property
    def parameters(self) -> FloatVector:
        if self.is_v_profile_copy:
            lst = [self.strength_scale]
            for i in range(self.n_profiles):
                lst.append(self.strength[i])
                lst.append(self.fwhm_v[i])
                lst.append(self.v_off[i])
        else:
            lst = [self.strength, self.fwhm_v, self.v_off]

        return np.array(lst, dtype=np.float64)

    @classmethod
    def _validate_is_prepared(cls, model: EmModel) -> None:
        if "evaluate_func" not in model.meta:
            raise ValueError(
                "Model instance was not properly prepared prior to "
                "error estimation! This could be hugely inefficient."
            )

    @classmethod
    @pydantic.validate_call
    def fromGaussianModel(cls, model: EmModel) -> Self:
        """
        Instantiates a GaussianSample from a GaussianModel or a velocity profile
        copy.
        """        
        cls._validate_is_prepared(model)
        
        if isinstance(model, GaussianModel):
            strength = model.strength.value
            fwhm_v = model.fwhm_v.value
            v_off = model.v_off.value

            wave = model.wave
            n_sigmas = model.n_sigmas
            name = model.name
            pure_name = model.pure_name

            is_v_profile_copy = False
            n_profiles = 1
            strength_scale = 1.0
        else:
            wave = model.wave
            n_sigmas = model.n_sigmas
            name = model.name
            pure_name = model.name
            
            is_v_profile_copy = True
            n_profiles = model.n_profiles
            strength_scale = model.strength_scale.value

            strength = np.empty(n_profiles, dtype=np.float64)
            fwhm_v = np.empty(n_profiles, dtype=np.float64)
            v_off = np.empty(n_profiles, dtype=np.float64)
            for i in range(n_profiles):
                strength[i] = getattr(model, f"strength_{i + 1}").value
                fwhm_v[i] = getattr(model, f"fwhm_v_{i + 1}").value
                v_off[i] = getattr(model, f"v_off_{i + 1}").value

        if model.dx:
            dx = model.dx
            sigma_res = None
            kwargs = {"wave": wave, "dx": dx}
        else:
            dx = None
            sigma_res = model.sigma_res
            kwargs = {"wave": wave, "sigma_res": sigma_res}

        linetype = model.linetype
        evaluate_func = choose_evaluate_func(dx is None)
        prime_func = choose_prime_func(dx is None)

        return cls(
            strength=strength,
            fwhm_v=fwhm_v,
            v_off=v_off,
            name=name,
            pure_name=pure_name,
            wave=wave,
            n_sigmas=n_sigmas,
            is_v_profile_copy=is_v_profile_copy,
            n_profiles=n_profiles,
            strength_scale=strength_scale,
            linetype=linetype,
            
            sigma_res=sigma_res,
            dx=dx,
            
            kwargs=kwargs,
            evaluate_func=evaluate_func,
            prime_func=prime_func,
        )

    @classmethod
    @pydantic.validate_call
    def fromParamVector(
        cls,
        param_vector: FloatVector,
        model: EmModel,
    ) -> Self:
        cls._validate_is_prepared(model)

        is_v_profile_copy = not isinstance(model, GaussianModel)
        if is_v_profile_copy:
            n_profiles = model.n_profiles
            strength_scale = param_vector[0]
            strength = param_vector[1:1+3*n_profiles:3]
            fwhm_v = param_vector[2:2+3*n_profiles:3]
            v_off = param_vector[3:3+3*n_profiles:3]
        else:
            n_profiles = 1
            strength_scale = 1.0
            strength = param_vector[0]
            fwhm_v = param_vector[1]
            v_off = param_vector[2]

        if model.dx:
            dx = model.dx
            sigma_res = None
            kwargs = {"wave": model.wave, "dx": dx}
        else:
            dx = None
            sigma_res = model.sigma_res
            kwargs = {"wave": model.wave, "sigma_res": sigma_res}

        evaluate_func = choose_evaluate_func(dx is None)
        prime_func = choose_prime_func(dx is None)

        return cls(
            strength=strength,
            fwhm_v=fwhm_v,
            v_off=v_off,
            name=model.name,
            pure_name=model.pure_name,
            wave=model.wave,
            n_sigmas=model.n_sigmas,
            is_v_profile_copy=is_v_profile_copy,
            n_profiles=n_profiles,
            strength_scale=strength_scale,
            
            dx=dx,
            sigma_res=sigma_res,

            kwargs=kwargs,
            evaluate_func=evaluate_func,
            prime_func=prime_func,
        )

    def __call__(
        self,
        x: float | FloatVector,
        y: FloatVector | None = None,
    ) -> float | FloatVector:
        if isinstance(x, (int, float)):
            assert y is None
            _x = np.array([x], dtype=np.float64)
            return self(_x, y=None)[0]

        if self.is_v_profile_copy:
            strengths = self.strength
            fwhm_vs = self.fwhm_v
            v_offs = self.v_off
        else:
            strengths = (self.strength,)
            fwhm_vs = (self.fwhm_v,)
            v_offs = (self.v_off,)

        for strength, fwhm_v, v_off in zip(strengths, fwhm_vs, v_offs):
            y = self.evaluate_func(
                x,
                self.strength_scale * strength, fwhm_v, v_off,
                **self.kwargs,
                y=y,
            )

        return y

    def prime(
        self, 
        x: float | FloatVector,
        y: FloatVector | None = None,
    ) -> float | FloatVector:
        if isinstance(x, (float, int)):
            assert y is None
            _x = np.array([x], dtype=np.float64)
            return self.prime(_x, y=None)[0]

        if self.is_v_profile_copy:
            strengths = self.strength
            fwhm_vs = self.fwhm_v
            v_offs = self.v_off
        else:
            strengths = (self.strength,)
            fwhm_vs = (self.fwhm_v,)
            v_offs = (self.v_off,)

        for strength, fwhm_v, v_off in zip(strengths, fwhm_vs, v_offs):
            y = self.prime_func(
                x,
                self.strength_scale * strength, fwhm_v, v_off,
                **self.kwargs,
                y=y,
            )

        return y

    @property
    def fwhm_v_c(self) -> float | FloatVector:
        """
        Intrinsic FWHM of the Gaussian function(s) in units of 'c'.
        """
        return self.fwhm_v / C_KMS

    @property
    def v_off_c(self) -> float | FloatVector:
        """
        Velocity offset(s) from the theoretical rest wavelength in units of 'c'.
        """
        return self.v_off / C_KMS

    @property
    def sorting_key(self) -> tuple[float, float, int]:
        return (4.0, self.wave, 0 if self.linetype == "n" else 1)

    @property
    def mu(self) -> float | FloatVector:
        """
        Observed wavelength(s) of the Gaussian peak(s).
        """
        return self.wave * (1 + self.v_off_c)

    @property
    def fwhm(self) -> float | FloatVector:
        """
        Wavelength FWHM of the Gaussian functions.
        """
        if self.dx is None:
            return self.mu * np.hypot(
                self.fwhm_v_c, 
                SIGMA_TO_FWHM * self.sigma_res,
            )
        return np.hypot(
            self.mu * self.fwhm_v_c,
            SIGMA_TO_FWHM * self.dx,
        )

    @property
    def sigma(self) -> float | FloatVector:
        """
        Wavelength dispersion of the Gaussian functions.
        """
        return self.fwhm * FWHM_TO_SIGMA

    @cached_property
    def peak(self) -> float:
        if self.is_v_profile_copy:
            msg = "Property 'peak' is not defined for velocity profile copies."
            raise NotImplementedError(msg)
        return GAUSS_AMP * self.strength / self.sigma

    @cached_property
    def dist(self) -> RVContinuousFrozen | list[RVContinuousFrozen]:
        if self.is_v_profile_copy:
            return [norm(*args) for args in zip(self.mu, self.sigma)]
        else:
            return norm(self.mu, self.sigma)

    @cached_property
    def pdf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        if self.is_v_profile_copy:
            ws = self.strength / self.strength.sum()
            ds = self.dist
        else:
            ws = (1.0,)
            ds = (self.dist,)

        def pdf(x: float | FloatVector, _ws=ws, _ds=ds) -> float | FloatVector:
            return sum(w * d.pdf(x) for w, d in zip(_ws, _ds))

        return pdf

    @cached_property
    def cdf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        if self.is_v_profile_copy:
            ws = self.strength / self.strength.sum()
            ds = self.dist
        else:
            ws = (1.0,)
            ds = (self.dist,)

        def cdf(x: float | FloatVector, _ws=ws, _ds=ds) -> float | FloatVector:
            return sum(w * d.cdf(x) for w, d in zip(_ws, _ds))

        return cdf

    @cached_property
    def isf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        if self.is_v_profile_copy and self.n_profiles > 1:
            msg = "Property 'isf' is not defined for multiple Gaussians."
            raise NotImplementedError(msg)

        _d = self.dist[0] if self.is_v_profile_copy else self.dist

        def isf(q: float | FloatVector, d=_d) -> float | FloatVector:
            return d.isf(q)

        return isf
