from typing import Self, Iterable, Literal, Callable
from numpy import array, float64
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from functools import cached_property
from pydantic import validate_call
from pydantic.dataclasses import dataclass

from quasar_models.line import GaussianModel, _VProfileCopy
from quasar_typing.numpy import FloatVector
from quasar_typing.astropy import Model_

from .gaussian_sample import GaussianSample
from ..utils.base_sample_list import BaseSampleList

@dataclass(frozen=True)
class GaussianSampleList(BaseSampleList[GaussianSample]):
    samples: list[GaussianSample]

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
    
    @property
    def model_type(self) -> Literal["em"]: 
        return "em"
    
    @cached_property
    def wave(self) -> FloatVector:
        return array([sample.wave for sample in self], dtype=float64)
    
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
    def sigma_v(self) -> FloatVector:
        sigma_v = []
        for sample in self:
            if sample.is_v_profile_copy:
                sigma_v.extend(sample.sigma_v)
            else:
                sigma_v.append(sample.sigma_v)
        return array(sigma_v, dtype=float64)
    
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
        if len(self) == 1:
            return self[0].cdf
        
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
        if len(self) > 1:
            msg = "Property 'isf' is not defined for multiple Gaussians."
            raise NotImplementedError(msg)
        
        return self[0].isf

    def sort(self, inplace: bool = False) -> Self:
        if inplace:
            return GaussianSampleList(self.samples).sort(inplace=True)
        
        self.samples = sorted(self.samples, key=lambda s: s.wave)
        return self

    @validate_call
    def splitAtWave(
        self,
        target_wave: float | set[float],
    ) -> tuple[Self | None, Self | None]:
        """
        ** PYDANTIC VALIDATED METHOD **

        Splits this GaussianSampleList into two: one containing samples whose 
        wave matches the target wave(s), and the other containing the rest.
        """
        sample_waves: set[float] = set(sample.wave for sample in self)
        in_dict: dict[float, bool] = (
            {w: (w < target_wave) for w in sample_waves}
            if isinstance(target_wave, float) else
            {w: (w in target_wave) for w in sample_waves}
        )

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
        ** PYDANTIC VALIDATED METHOD **
        
        Splits this GaussianSampleList into two: one containing samples whose 
        line matches the target line(s), and the other containing the rest.
        """
        
        sample_lines: set[str] = set(sample.pure_name for sample in self)
        in_dict: dict[str, bool] = (
            {line: (line == target_line) for line in sample_lines}
            if isinstance(target_line, str) else
            {line: (line in target_line) for line in sample_lines}
        )
        
        targets = GaussianSampleList.fromGaussianSamples(
            sample for sample in self if in_dict[sample.pure_name]
        )
        background = GaussianSampleList.fromGaussianSamples(
            sample for sample in self if not in_dict[sample.pure_name]
        )

        return targets or None, background or None