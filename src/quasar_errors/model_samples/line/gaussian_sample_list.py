from typing import Self, Iterable, Literal, Callable, ClassVar
from numpy import array, float64, fromiter
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from functools import cached_property
from pydantic.dataclasses import dataclass

from quasar_models.line import GaussianModel, _VProfileCopy
from quasar_typing.numpy import FloatVector
from quasar_typing.astropy import Model_
from quasar_utils.decorators import validate_call

from .gaussian_sample import GaussianSample
from ..utils.base_sample_list import BaseSampleList

@dataclass(frozen=True)
class GaussianSampleList(BaseSampleList[GaussianSample]):
    samples: list[GaussianSample]

    model_type: ClassVar[Literal["em"]] = "em"

    @classmethod
    @validate_call
    def fromGaussianSamples(
        cls,
        samples: Iterable[GaussianSample],
    ) -> Self:
        return GaussianSampleList(list(samples))
    
    @classmethod
    @validate_call
    def fromGaussianSubmodels(
        cls,
        submodels: Iterable[GaussianModel | _VProfileCopy],
    ) -> Self:
        return cls.fromGaussianSamples(
            GaussianSample.fromGaussianModel.__wrapped__(GaussianSample, model)
            for model in submodels
        )
    
    @classmethod
    @validate_call
    def fromGaussianModels(
        cls,
        model: Model_
    ) -> Self:
        return cls.fromGaussianSubmodels.__wrapped__(
            cls,
            filter(
                lambda m: isinstance(m, (GaussianModel, _VProfileCopy)),
                (model,) if model.n_submodels == 1 else model,
            )
        )
        
    @cached_property
    def wave(self) -> FloatVector:
        return fromiter((sample.wave for sample in self), dtype=float64)
    
    @cached_property
    def strength(self) -> FloatVector:
        strength = []
        for sample in self:
            if sample.is_v_profile_copy:
                strength.extend(sample.strength)
            else:
                strength.append(sample.strength)
        return array(strength, dtype=float64)
    
    @cached_property
    def weight(self) -> FloatVector:
        return self.strength / self.strength.sum()
    
    @cached_property
    def fwhm_v(self) -> FloatVector:
        fwhm_v = []
        for sample in self:
            if sample.is_v_profile_copy:
                fwhm_v.extend(sample.fwhm_v)
            else:
                fwhm_v.append(sample.fwhm_v)
        return array(fwhm_v, dtype=float64)
    
    @cached_property
    def v_off(self) -> FloatVector:
        v_off = []
        for sample in self:
            if sample.is_v_profile_copy:
                v_off.extend(sample.v_off)
            else:
                v_off.append(sample.v_off)
        return array(v_off, dtype=float64)
    
    @cached_property
    def mu(self) -> FloatVector:
        mu = []
        for sample in self:
            if sample.is_v_profile_copy:
                mu.extend(sample.mu)
            else:
                mu.append(sample.mu)
        return array(mu, dtype=float64)
    
    @cached_property
    def fwhm(self) -> FloatVector:
        fwhm = []
        for sample in self:
            if sample.is_v_profile_copy:
                fwhm.extend(sample.fwhm)
            else:
                fwhm.append(sample.fwhm)
        return array(fwhm, dtype=float64)
    
    @cached_property
    def sigma(self) -> FloatVector:
        sigma = []
        for sample in self:
            if sample.is_v_profile_copy:
                sigma.extend(sample.sigma)
            else:
                sigma.append(sample.sigma)
        return array(sigma, dtype=float64)
    
    @cached_property
    def peak(self) -> float:
        if len(self) > 1:
            msg = "Property 'peak' is not defined for multiple Gaussians."
            raise NotImplementedError(msg)
        return self[0].peak
    
    @cached_property
    def dist(self) -> list[rv_continuous_frozen]:
        out = []
        for sample in self:
            if sample.is_v_profile_copy:
                out.extend(sample.dist)
            else:
                out.append(sample.dist)
        return out
    
    @cached_property
    def pdf(self) -> Callable[[float | FloatVector], float | FloatVector]:
        if len(self) == 1:
            return self[0].pdf
        
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
        if len(self) == 1:
            return self[0].cdf
        
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
        if len(self) > 1:
            msg = "Property 'isf' is not defined for multiple Gaussians."
            raise NotImplementedError(msg)
        
        return self[0].isf

    def sort(self, inplace: bool = False) -> Self:
        if not inplace:
            return GaussianSampleList(self.samples).sort(inplace=True)
        
        self.samples = sorted(self.samples, key=lambda s: s.wave)
        return self

    @validate_call
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

        sample_waves = set(sample.wave for sample in self)
        in_dict = {w: (w in target_wave) for w in sample_waves}

        targets = GaussianSampleList.fromGaussianSamples(
            sample for sample in self if in_dict[sample.wave]
        )
        background = GaussianSampleList.fromGaussianSamples(
            sample for sample in self if not in_dict[sample.wave]
        )

        return targets or None, background or None
    
    @validate_call
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
        
        sample_lines = set(sample.pure_name for sample in self)
        in_dict = {line: (line in target_line) for line in sample_lines}
        
        targets = GaussianSampleList.fromGaussianSamples(
            sample for sample in self if in_dict[sample.pure_name]
        )
        background = GaussianSampleList.fromGaussianSamples(
            sample for sample in self if not in_dict[sample.pure_name]
        )

        return targets or None, background or None