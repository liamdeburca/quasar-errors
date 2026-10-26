__all__ = [
    "get_intercept",
    "get_peak",
    "get_quantile",
    "get_width_at_height",
    "u_xinterp",
    "u_yinterp",
]
import math
from collections.abc import Iterable

from numpy import (
    append,
    array,
    cumsum,
    dot,
    float64,
    interp,
    nan_to_num,
    searchsorted,
)
from quasar_typing.literals import FWHMStrategy
from quasar_typing.numpy import FloatVector

from ..utils import mirror_and_rebin
from .approx import get_intercept as cy_get_intercept
from .approx import get_peak as cy_get_peak

# from .approx import get_quantile as cy_get_quantile
from .approx import u_xinterp as cy_u_xinterp
from .approx import u_yinterp as cy_u_yinterp


def u_yinterp(
    x: float,
    xp: FloatVector,
    yp: FloatVector,
    dyp: FloatVector,
) -> tuple[float, float]:
    assert xp.size >= 2, xp.size
    i = max(searchsorted(xp, x, side="right") - 1, 0)
    return cy_u_yinterp(x, xp[i : i + 2], yp[i : i + 2], dyp[i : i + 2])


def u_xinterp(
    y: float,
    xp: FloatVector,
    yp: FloatVector,
    dyp: FloatVector,
) -> tuple[float, float]:
    assert xp.size >= 2, xp.size
    i = max(searchsorted(yp, y, side="right") - 1, 0)
    return cy_u_xinterp(y, xp[i : i + 2], yp[i : i + 2], dyp[i : i + 2])


def get_intercept(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    y_crit: float,
    strategy: FWHMStrategy,
) -> tuple[float, float]:
    assert x.size == y.size == dy.size
    _strategy = {
        "narrowest": 0,
        "average": 1,
        "widest": 2,
    }.get(strategy, 0)
    return cy_get_intercept(x, y, dy, y_crit, _strategy)


def get_left_edge_at_height(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    x_max: float,
    y_crit: float,
    strategy: FWHMStrategy,
) -> tuple[float, float]:
    assert x.size == y.size
    assert x.size == dy.size
    mode_idx = searchsorted(x, x_max, side="left")

    val = x[0]
    unc = 0.0
    if mode_idx > 0:
        val, unc = get_intercept(
            x[:mode_idx],
            y[:mode_idx] - y_crit,
            dy[:mode_idx],
            0.0,
            strategy,
        )
    return val, unc


def get_right_edge_at_height(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    x_max: float,
    y_crit: float,
    strategy: FWHMStrategy,
) -> tuple[float, float]:
    assert x.size == y.size == dy.size
    mode_idx = searchsorted(x, x_max, side="right") - 1

    val = x[-1]
    unc = 0.0
    if mode_idx < x.size - 1:
        val, unc = get_intercept(
            x[mode_idx:],
            y_crit - y[mode_idx:],
            dy[mode_idx:],
            0.0,
            strategy,
        )
    return val, unc


def get_width_at_height(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    x_max: float,
    y_crit: float,
    strategy: FWHMStrategy,
) -> tuple[float, float]:
    assert x.size == y.size == dy.size
    left_val, left_unc = get_left_edge_at_height(
        x, y, dy, x_max, y_crit, strategy,
    )
    right_val, right_unc = get_right_edge_at_height(
        x, y, dy, x_max, y_crit, strategy,
    )
    val = right_val - left_val
    unc = math.hypot(left_unc, right_unc)
    
    return val, unc


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

    cdf = append(0, cumsum(nan_to_num(y)))
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    out = interp(qs, cdf, edges)
    return out[0] if isinstance(q, float) else out


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

    return cy_get_peak(x, y, edges)


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
