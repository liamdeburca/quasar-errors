__all__ = [
    'get_intercept',
    'get_width_at_height',
    'get_quantile',
    'get_peak',
    'get_median_absolute_deviation',
    'get_average_absolute_deviation',
]

from typing import Iterable
from numpy import invert, argwhere, interp, append, dot, hypot
from quasar_typing.numpy import FloatVector
from quasar_typing.misc import FWHMStrategy

from numba import njit

from ..utils import mirror_and_rebin

@njit(fastmath=True)
def u_yinterp(
    x: float,
    xp: Iterable[float],
    yp: Iterable[float],
    dyp: Iterable[float],
) -> tuple[float, float]:
    """
    ** NUMBA OPTIMISED FUNCTION **

    Interpolates the y-value at 'x' using the reference points 'xp' and 'yp', and 
    also propagates the uncertainty 'dyp' associated with 'yp'.
    """
    dx = xp[1] - xp[0]
    f0 = (xp[1] - x) / dx
    f1 = (x - xp[0]) / dx

    return (
        f0 * yp[0] + f1 * yp[1],
        hypot(f0 * dyp[0], f1 * dyp[1]),
    )

@njit(fastmath=True)
def u_xinterp(
    y: float,
    xp: Iterable[float],
    yp: Iterable[float],
    dyp: Iterable[float],
) -> tuple[float, float]:
    """
    ** NUMBA OPTIMISED FUNCTION **

    Interpolates the x-value at 'y' using the reference points 'xp' and 'yp', and 
    also propagates the uncertainty 'dyp' associated with 'yp'.
    """
    dx = xp[1] - xp[0]
    dy = yp[1] - yp[0]
    f0 = (yp[1] - y) / dy
    f1 = (y - yp[0]) / dy

    return (
        f0 * xp[0] + f1 * xp[1],
        abs(dx) / abs(dy) * hypot(f0 * dyp[0], f1 * dyp[1]),
    )

@njit(fastmath=True)
def get_intercept(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    y_crit: float,
    strategy: FWHMStrategy,
) -> tuple[float, float]:
    """
    ** NUMBA OPTIMISED FUNCTION **

    Calculates the x-value where the curve defined by (x, y) intersects the 
    horizontal line at 'y_crit', and also propagates the uncertainty in this 
    x-value based on 'dy'.
    """
    below = (y < y_crit)
    if below.all():
        return (x[-1], dy[-1])
    elif not below.any():
        return (x[0], dy[0])
    
    above = invert(below)
    indices = argwhere(below[:-1] & above[1:]).flatten()

    if indices.size > 1 and strategy == 'average':
        idx_r = indices[0]
        left = u_xinterp(
            y_crit, 
            x[idx_r:idx_r+2],
            y[idx_r:idx_r+2],
            dy[idx_r:idx_r+2],
        )
        idx_l = indices[-1]
        right = u_xinterp(
            y_crit, 
            x[idx_l:idx_l+2],
            y[idx_l:idx_l+2],
            dy[idx_l:idx_l+2],
        )

        return (
            (left[0] + right[0]) / 2,
            hypot(left[1], right[1]) / 2,
        )

    if indices.size == 1 or strategy == 'widest':
        idx = indices[0]
    else:
        idx = indices[-1]

    return u_xinterp(
        y_crit, 
        x[idx:idx+2],
        y[idx:idx+2],
        dy[idx:idx+2],
    )

@njit(fastmath=True)
def get_width_at_height(
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    x_max: float,
    y_crit: float,
    strategy: FWHMStrategy,
) -> tuple[float, float]:
    """
    ** NUMBA OPTIMISED FUNCTION **

    Calculates the width of the curve defined by (x, y) at the height 'y_crit',
    and also propagates the uncertainty in this width based on 'dy'.
    """
    is_left = (x < x_max)
    is_right = invert(is_left)

    left_edge = (x[0], 0)
    if is_left.any():
        left_edge = get_intercept(
            x[is_left],
            y[is_left],
            dy[is_left],
            y_crit,
            strategy=strategy,
        )

    right_edge = (x[-1], 0)
    if is_right.any():
        right_edge = get_intercept(
            x[is_right][::-1],
            y[is_right][::-1],
            dy[is_right][::-1],
            y_crit,
            strategy=strategy,
        )

    return (
        right_edge[0] - left_edge[0],
        hypot(left_edge[1], right_edge[1]),
    )

@njit(fastmath=True)
def get_quantile(
    q: float | Iterable[float],
    x: FloatVector,
    y: FloatVector,
    dx: float | FloatVector,
) -> float | FloatVector:
    """
    ** NUMBA OPTIMISED FUNCTION **
    """
    edges = append(
        x - dx / 2,
        x[-1] + (dx if isinstance(dx, float) else dx[-1]) / 2,
    )
    cdf = append(0, y.cumsum())
    cdf /= cdf[-1]
    return interp(q, cdf, edges)

@njit(fastmath=True)
def get_peak(
    x: FloatVector,
    y: FloatVector,
    dx: float | FloatVector,
) -> tuple[float, float]:
    """
    ** NUMBA OPTIMISED FUNCTION **
    """
    edges = append(
        x - dx / 2,
        x[-1] + (dx if isinstance(dx, float) else dx[-1]) / 2,
    )
    idx = y.argmax().flatten()[0]
    y_max = y[idx]

    if idx == 0 or idx == x.size -1:
        x_max = x[idx]

    if not ((idx == 0) or (idx == x.size - 1)):
        x_l, x_r = edges[idx:idx+2]
        y_l, _, y_r = y[idx-1:idx+2]
        x_max = x_l + (x_r - x_l) * ((y_max - y_l) / (2 * y_max - y_l - y_r))

    return (x_max, y_max)

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