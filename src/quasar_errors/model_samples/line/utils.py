from dataclasses import dataclass
from typing import TypeVar

from quasar_typing.numpy import FloatVector
from scipy.stats._distn_infrastructure import rv_continuous_frozen

T = TypeVar("T", bound=float | FloatVector)

@dataclass
class CombinedDist:
    dists: list[rv_continuous_frozen]
    weights: list[float]

    def pdf[T](self, x: T) -> T:
        if len(self.dists) == 1:
            return self.dists[0].pdf(x)
        return sum(w * d.pdf(x) for w, d in zip(self.weights, self.dists))

    def cdf[T](self, x: T) -> T:
        if len(self.dists) == 1:
            return self.dists[0].cdf(x)
        return sum(w * d.cdf(x) for w, d in zip(self.weights, self.dists))

    def isf[T](self, x: T) -> T:
        if len(self.dists) == 1:
            return self.dists[0].isf(x)
        raise NotImplementedError

    
