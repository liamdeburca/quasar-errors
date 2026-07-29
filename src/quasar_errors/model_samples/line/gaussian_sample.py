from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from math import log, pi
from typing import ClassVar, Literal, Self, Union

from numpy import array, empty, float64, hypot, zeros_like
from quasar_models._core.modeling.gaussian import (
    GaussianEvaluate,
    evaluate_v,
)
from quasar_models.line import GaussianModel, _VProfileCopy
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import rv_continuous_frozen_
from quasar_utils.decorators import validate_call
from scipy.stats import norm

from ..utils.base_sample import BaseSample

GAUSS_AMP = 1 / (2 * pi) ** 0.5
SIGMA_TO_FWHM = 2 * (2 * log(2)) ** 0.5
FWHM_TO_SIGMA = 1 / SIGMA_TO_FWHM


@dataclass(frozen=True, kw_only=True)
class GaussianSample(BaseSample):
    strength: float | FloatVector
    fwhm_v: float | FloatVector
    v_off: float | FloatVector

    name: str
    pure_name: str
    wave: float
    sigma_res: float
    n_sigmas: float
    is_v_profile_copy: bool
    n_profiles: int
    strength_scale: float

    evaluate_func: GaussianEvaluate = evaluate_v

    model_type: ClassVar[Literal["em"]] = "em"

    @classmethod
    @validate_call
    def fromGaussianModel(
        cls,
        model: Union[GaussianModel, _VProfileCopy],
    ) -> Self:
        """
        Instantiates a GaussianSample from a GaussianModel or a velocity profile
        copy.
        """
        if isinstance(model, GaussianModel):
            strength = model.strength.value
            fwhm_v = model.fwhm_v.value
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
            wave = model.wave
            sigma_res = model.sigma_res
            n_sigmas = model.n_sigmas
            name = pure_name = model.name
            is_v_profile_copy = True
            n_profiles = model.n_profiles
            strength_scale = model.strength_scale.value

            strength = empty(n_profiles, dtype=float64)
            fwhm_v = empty(n_profiles, dtype=float64)
            v_off = empty(n_profiles, dtype=float64)
            for i in range(n_profiles):
                strength[i] = getattr(model, f"strength_{i + 1}").value
                fwhm_v[i] = getattr(model, f"fwhm_v_{i + 1}").value
                v_off[i] = getattr(model, f"v_off_{i + 1}").value

        else:
            raise TypeError("Model must be either GaussianModel or velocity profile copy.")

        return GaussianSample(
            strength=strength,
            fwhm_v=fwhm_v,
            v_off=v_off,
            name=name,
            pure_name=pure_name,
            wave=wave,
            sigma_res=sigma_res,
            n_sigmas=n_sigmas,
            is_v_profile_copy=is_v_profile_copy,
            n_profiles=n_profiles,
            strength_scale=strength_scale,
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

        if self.is_v_profile_copy:
            strengths = self.strength
            fwhm_vs = self.fwhm_v
            v_offs = self.v_off
        else:
            strengths = (self.strength,)
            fwhm_vs = (self.fwhm_v,)
            v_offs = (self.v_off,)

        if y is None:
            y = zeros_like(x, dtype=float64)

        for strength, fwhm_v, v_off in zip(strengths, fwhm_vs, v_offs):
            self.evaluate_func(
                x,
                self.strength_scale * strength, fwhm_v, v_off,
                wave=self.wave, sigma_res=self.sigma_res,
                y=y,
            )

        return y

    def prime(self, x: float | FloatVector) -> float | FloatVector:
        raise NotImplementedError()
        # if isinstance(x, (float, int)):
        #     _x = array([x], dtype=float64)
        #     return self.prime(_x)[0]

        # if self.is_v_profile_copy:
        #     strengths = self.strength
        #     fwhm_vs = self.fwhm_v
        #     v_offs = self.v_off
        # else:
        #     strengths = (self.strength,)
        #     fwhm_vs = (self.fwhm_v,)
        #     v_offs = (self.v_off,)

        # y = zeros_like(x, dtype=float64)
        # for strength, fwhm_v, v_off in zip(strengths, fwhm_vs, v_offs):
        #     prime_v(
        #         x,
        #         strength,
        #         fwhm_v,
        #         v_off,
        #         wave=self.wave,
        #         sigma_res=self.sigma_res,
        #         y=y,
        #     )
        # y *= self.strength_scale
        # return y

    @property
    def sorting_key(self) -> tuple[float, float]:
        return (4.0, self.wave)

    @cached_property
    def mu(self) -> float | FloatVector:
        return self.wave * (1 + self.v_off)

    @cached_property
    def fwhm(self) -> float | FloatVector:
        return self.mu * hypot(self.fwhm_v, SIGMA_TO_FWHM * self.sigma_res)

    @property
    def sigma(self) -> float | FloatVector:
        return self.fwhm * FWHM_TO_SIGMA

    @cached_property
    def peak(self) -> float:
        if self.is_v_profile_copy:
            msg = "Property 'peak' is not defined for velocity profile copies."
            raise NotImplementedError(msg)
        return GAUSS_AMP * self.strength / self.sigma

    @cached_property
    def dist(self) -> rv_continuous_frozen_ | list[rv_continuous_frozen_]:
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

        def pdf(x: float | FloatVector) -> float | FloatVector:
            nonlocal ws, ds
            return sum(w * d.pdf(x) for w, d in zip(ws, ds))

        return pdf

    @cached_property
    def cdf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        if self.is_v_profile_copy:
            ws = self.strength / self.strength.sum()
            ds = self.dist
        else:
            ws = (1.0,)
            ds = (self.dist,)

        def cdf(x: float | FloatVector) -> float | FloatVector:
            nonlocal ws, ds
            return sum(w * d.cdf(x) for w, d in zip(ws, ds))

        return cdf

    @cached_property
    def isf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        if self.is_v_profile_copy and self.n_profiles > 1:
            msg = "Property 'isf' is not defined for multiple Gaussians."
            raise NotImplementedError(msg)

        dist = self.dist[0] if self.is_v_profile_copy else self.dist

        def isf(q: float | FloatVector) -> float | FloatVector:
            nonlocal dist
            return dist.isf(q)

        return isf
