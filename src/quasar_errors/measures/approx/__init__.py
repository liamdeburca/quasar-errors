__all__ = [
    "u_yinterp",
    "u_xinterp",
    "get_intercept",
    "get_width_at_height",
    "get_quantile",
    "get_peak",
]

from . import approx
from ..utils import mirror_and_rebin

from typing import Iterable
from numpy import array, append, float64, empty, interp, dot
from quasar_typing.numpy import FloatVector
from quasar_typing.misc import FWHMStrategy

from quasar_utils.decorators import validate_call

@validate_call
def u_yinterp(
    x: float,
    xp: FloatVector,
    yp: FloatVector,
    dyp: FloatVector,
) -> tuple[float, float]:
    assert xp.size == yp.size == dyp.size == 2
    return approx.u_yinterp(x, xp, yp, dyp)

@validate_call
def u_xinterp(
    y: float,
    xp: FloatVector,
    yp: FloatVector,
    dyp: FloatVector,
) -> tuple[float, float]:
    assert xp.size == yp.size == dyp.size == 2
    return approx.u_xinterp(y, xp, yp, dyp)

@validate_call
def get_intercept(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    y_crit: float,
    strategy: FWHMStrategy,
) -> tuple[float, float]:
    assert x.size == y.size == dy.size
    _strategy = {
        'narrowest': 0,
        'average': 1,
        'widest': 2,
    }.get(strategy, 0)
    return approx.get_intercept(x, y, dy, y_crit, _strategy)

@validate_call
def get_width_at_height(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    x_max: float,
    y_crit: float,
    strategy: FWHMStrategy,
) -> tuple[float, float]:
    assert x.size == y.size == dy.size
    _strategy = {
        'narrowest': 0,
        'average': 1,
        'widest': 2,
    }.get(strategy, 0)
    return approx.get_width_at_height(x, y, dy, x_max, y_crit, _strategy)

@validate_call
def get_quantile(
    q: float | Iterable[float],
    x: FloatVector,
    y: FloatVector,
    dx: float | FloatVector,
) -> float | FloatVector:
    qs = array([q] if isinstance(q, float) else q, dtype=float64)
    if isinstance(dx, float):
        assert x.size == y.size
        edges = append(x - dx / 2, x[-1] + dx / 2)
    else:
        assert x.size == y.size == dx.size
        edges = append(x - dx / 2, x[-1] + dx[-1] / 2)

    cdf = empty(x.size + 1, dtype=float64)
    out = empty(qs.size, dtype=float64)
    _ = approx.get_quantile(qs, x, y, edges, cdf, out)

    return out[0] if isinstance(q, float) else out

@validate_call
def get_peak(
    x: FloatVector,
    y: FloatVector,
    dx: float | FloatVector,
) -> tuple[float, float]:
    if isinstance(dx, float):
        assert x.size == y.size
        edges = append(x - dx / 2, x[-1] + dx / 2)
    else:
        assert x.size == y.size == dx.size
        edges = append(x - dx / 2, x[-1] + dx[-1] / 2)

    return approx.get_peak(x, y, edges)

@validate_call
def get_median_absolute_deviation(
    cen: float,
    x: FloatVector,
    y: FloatVector,
    dx: float | FloatVector,
) -> float:
    edges, vals = mirror_and_rebin(cen, x, y, dx)
    cdf = append(0, vals.cumsum())
    cdf /= cdf[-1]
    return interp(0.5, cdf, edges)

@validate_call
def get_average_absolute_deviation(
    cen: float,
    x: FloatVector,
    y: FloatVector,
    dx: float | FloatVector,
) -> float:
    edges, vals = mirror_and_rebin(cen, x, y, dx)
    x_new = 0.5 * (edges[:-1] + edges[1:])
    W = vals.sum()
    return dot(x_new, vals) / W if W > 0 else 0