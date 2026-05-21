from typing import Callable, Self
from dataclasses import field
from pydantic.dataclasses import dataclass
from numpy import clip, histogram, inf, isfinite
from scipy.stats import uniform, loguniform, norm
from astropy.modeling import Parameter

from quasar_typing.numpy import FloatVector
from quasar_typing.bounds import AstropyBounds
from quasar_typing.scipy import rv_continuous_frozen_, rv_histogram_

@dataclass
class PseudoParam:
    """
    Lightweight class mimicking an AstroPy Parameter.
    """
    value: float
    bounds: AstropyBounds
    fixed: bool
    tied: Callable | bool | None

    _uniform: rv_continuous_frozen_ | None = field(default=None, repr=False, kw_only=True)
    _loguniform: rv_continuous_frozen_ | None = field(default=None, repr=False, kw_only=True)

    _hist: rv_histogram_ | None = field(default=None, repr=False, kw_only=True)
    _norm: rv_continuous_frozen_ | None = field(default=None, repr=False, kw_only=True)

    def __post_init__(self):
        self.bounds = (
            -inf if self.bounds[0] is None else self.bounds[0],
            inf if self.bounds[1] is None else self.bounds[1],
        )
        if (self._uniform is None) and isfinite(self.bounds).all():
            self._uniform = uniform(
                loc=self.bounds[0], 
                scale=self.bounds[1] - self.bounds[0],
            )
            if (self._loguniform is None) and all(b > 0 for b in self.bounds):
                self._loguniform = loguniform(*self.bounds)

    def __str__(self) -> str:
        return "<PseudoParam({}, bounds={}, fixed={}, tied={})>".format(
            self.value, 
            self.bounds,
            self.fixed, 
            self.tied is not None,
        )
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __getstate__(self) -> dict:
        return {
            'value': self.value,
            'bounds': self.bounds,
            'fixed': self.fixed,
            'tied': self.tied,
            '_uniform': self._uniform,
            '_loguniform': self._loguniform,
            '_hist': self._hist,
            '_norm': self._norm,
        }
    
    def __setstate__(self, state: dict) -> None:
        self.value = state['value']
        self.bounds = state['bounds']
        self.fixed = state['fixed']
        self.tied = state['tied']
        self._uniform = state['_uniform']
        self._log_uniform = state['_log_uniform']
        self._hist = state['_hist']
        self._norm = state['_norm']

    @classmethod
    def from_parameter(cls, param: Parameter) -> Self:
        return PseudoParam(
            param.value,
            param.bounds,
            param.fixed,
            param.tied if callable(param.tied) else None,
        )
    
    @property
    def free(self) -> bool:
        return (not self.fixed) and (self.tied is None)

    def uniform(self, u: float | FloatVector) -> float | FloatVector:
        return self._uniform.isf(u)
    
    def hist(self, u: float | FloatVector) -> float | FloatVector:
        return self._hist.isf(u)
    
    def norm(self, u: float | FloatVector) -> float | FloatVector:
        return clip(
            self._norm.isf(u),
            self.bounds[0] if self.bounds[0] is not None else -inf,
            self.bounds[1] if self.bounds[1] is not None else inf,
        )
    
    def learn(
        self, 
        samples: FloatVector,
        bins: int | FloatVector = 10,
        weights: FloatVector | None = None,
    ) -> None:
        hist = histogram(
            samples, 
            bins=bins, 
            range=self.bounds, 
            weights=weights, 
            density=False,
        )

        self._dist = rv_histogram_(hist)
        self._norm = norm(loc=self._dist.mean(), scale=self._dist.std())

    def update_bounds_from_samples(
        self,
        samples: FloatVector,
        scale: float = 1.0,
    ) -> None:
        lb, ub = self.bounds
        
        if lb is not None:
            lb = max(lb, self.value + scale * (samples.min() - self.value))
        if ub is not None:
            ub = min(ub, self.value + scale * (samples.max() - self.value))
        
        self.bounds = (lb, ub)