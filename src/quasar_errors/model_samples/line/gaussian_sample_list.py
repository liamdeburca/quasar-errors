import re
from collections.abc import Callable, Iterable
from typing import Self

import numpy as np
import pydantic
from quasar_models import EmModel
from quasar_typing.astropy import ModelType
from quasar_typing.numpy import FloatVector
from quasar_typing.re import StrPattern
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from ..utils.base_sample_list import BaseSampleList
from .gaussian_sample import C_KMS, FWHM_TO_SIGMA, GaussianSample
from .utils import CombinedDist


class GaussianSampleList(BaseSampleList[GaussianSample]):
    samples: list[GaussianSample]


    @classmethod
    @pydantic.validate_call
    def fromGaussianSamples(
        cls,
        samples: Iterable[GaussianSample],
    ) -> Self:
        return cls(samples=sorted(samples, key=lambda s: s.sorting_key))

    @classmethod
    @pydantic.validate_call
    def fromGaussianSubmodels(
        cls,
        submodels: Iterable[EmModel],
    ) -> Self:
        return cls.fromGaussianSamples(
            GaussianSample.fromGaussianModel(model)
            for model in sorted(submodels, key=lambda m: m.sorting_key)
        )

    @classmethod
    @pydantic.validate_call
    def fromGaussianModels(cls, model: ModelType) -> Self:
        return cls.fromGaussianSubmodels(
            filter(
                lambda m: getattr(m, "model_type", None) == "em",
                (model,) if model.n_submodels == 1 else model,
            ),
        )

    @property
    def wave(self) -> FloatVector:
        return np.fromiter((sample.wave for sample in self.samples), dtype=np.float64)

    @property
    def strength(self) -> FloatVector:
        strength = []
        for sample in self.samples:
            if sample.is_v_profile_copy:
                strength.extend(sample.strength)
            else:
                strength.append(sample.strength)
        return np.array(strength, dtype=np.float64)

    @property
    def weight(self) -> FloatVector:
        return self.strength / self.strength.sum()

    @property
    def fwhm_v(self) -> FloatVector:
        fwhm_v = []
        for sample in self.samples:
            if sample.is_v_profile_copy:
                fwhm_v.extend(sample.fwhm_v)
            else:
                fwhm_v.append(sample.fwhm_v)
        return np.array(fwhm_v, dtype=np.float64)

    @property
    def fwhm_v_c(self) -> FloatVector:
        return self.fwhm_v / C_KMS

    @property
    def v_off(self) -> FloatVector:
        v_off = []
        for sample in self.samples:
            if sample.is_v_profile_copy:
                v_off.extend(sample.v_off)
            else:
                v_off.append(sample.v_off)
        return np.array(v_off, dtype=np.float64)

    @property
    def v_off_c(self) -> FloatVector:
        return self.v_off / C_KMS

    @property
    def mu(self) -> FloatVector:
        mu = []
        for sample in self.samples:
            if sample.is_v_profile_copy:
                mu.extend(sample.mu)
            else:
                mu.append(sample.mu)
        return np.array(mu, dtype=np.float64)

    @property
    def fwhm(self) -> FloatVector:
        fwhm = []
        for sample in self.samples:
            if sample.is_v_profile_copy:
                fwhm.extend(sample.fwhm)
            else:
                fwhm.append(sample.fwhm)
        return np.array(fwhm, dtype=np.float64)

    @property
    def sigma(self) -> FloatVector:
        return self.fwhm * FWHM_TO_SIGMA

    @property
    def peak(self) -> float:
        if len(self.samples) > 1:
            msg = "Property 'peak' is not defined for multiple Gaussians."
            raise NotImplementedError(msg)
        return self.samples[0].peak

    @property
    def dist(self) -> list[rv_continuous_frozen]:
        out = []
        for sample in self.samples:
            if sample.is_v_profile_copy:
                out.extend(sample.dist)
            else:
                out.append(sample.dist)
        return out

    @property
    def _combined_dist(self) -> CombinedDist:
        return CombinedDist(dists=self.dist, weights=self.weight)

    @property
    def pdf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        return self._combined_dist.pdf

    @property
    def cdf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        return self._combined_dist.cdf

    @property
    def isf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        if len(self.dist) > 1:
            raise NotImplementedError
        return self._combined_dist.isf

    @pydantic.validate_call
    def splitAtPattern(
        self,
        pattern: StrPattern | tuple[StrPattern, ...] | Callable[[str], bool],
    ) -> tuple[Self | None, Self | None]:

        func: Callable[[str], bool]
        if isinstance(pattern, str):
            func = lambda c: re.match(pattern, c) is not None
        elif isinstance(pattern, tuple):
            func = lambda c: any(
                re.match(p, c) is not None
                for p in pattern
            )
        else:
            func = pattern

        t: list[GaussianSample] = []
        b: list[GaussianSample] = []
        for sample in self.samples:
            is_match = func(sample.name)
            if is_match:
                t.append(sample)
            else:
                b.append(sample)

        targets = GaussianSampleList.fromGaussianSamples(t) if t else None
        background = GaussianSampleList.fromGaussianSamples(b) if b else None

        return targets, background

    @pydantic.validate_call
    def splitAtWave(
        self,
        target_wave: float | set[float],
    ) -> tuple[Self | None, Self | None]:
        """
        Splits this GaussianSampleList into two: one containing samples whose
        wave matches the target wave(s), and the other containing the rest.
        """
        if not isinstance(target_wave, set):
            target_wave = {target_wave}

        sample_waves = {sample.wave for sample in self.samples}
        in_dict = {w: (w in target_wave) for w in sample_waves}

        targets = GaussianSampleList.fromGaussianSamples(
            sample for sample in self.samples if in_dict[sample.wave]
        )
        background = GaussianSampleList.fromGaussianSamples(
            sample for sample in self.samples if not in_dict[sample.wave]
        )

        return targets or None, background or None

    @pydantic.validate_call
    def splitAtLine(
        self,
        target_line: str | set[str],
    ) -> tuple[Self | None, Self | None]:
        """
        Splits this GaussianSampleList into two: one containing samples whose
        line matches the target line(s), and the other containing the rest.
        """
        if not isinstance(target_line, set):
            target_line = {target_line}

        tg_samples = []
        bg_samples = []
        for sample in self.samples:
            ls = tg_samples if (sample.pure_name in target_line) else bg_samples
            ls.append(sample)

        targets = GaussianSampleList.fromGaussianSamples(tg_samples) \
            if tg_samples \
            else None
        background = GaussianSampleList.fromGaussianSamples(bg_samples) \
            if bg_samples \
            else None

        return targets, background

    def prime(self, x: float | FloatVector) -> float | FloatVector:
        if isinstance(x, (float, int)):
            _x = np.array([x], np.float64)
            return self.prime(_x)[0]

        y = np.zeros_like(x)
        for em_sample in self.samples:
            y += em_sample.prime(x)

        return y
