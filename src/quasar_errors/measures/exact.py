__all__ = [
    "get_average_absolute_deviation",
    "get_intercept",
    "get_median_absolute_deviation",
    "get_peak",
    "get_quantile",
    "get_width_at_height",
]

from collections.abc import Callable, Iterable
from typing import Literal

from numpy import argwhere, array, dot, float64, invert
from quasar_typing.misc import FWHMStrategy
from quasar_typing.numpy import FloatVector
from scipy.optimize import minimize_scalar, newton
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from ..model_samples import GaussianSample, GaussianSampleList


def get_intercept(
    func: Callable[[FloatVector], FloatVector],
    x: FloatVector,
    y_crit: float,
    origin: Literal["left", "right"],
    strategy: FWHMStrategy,
    *,
    fprime: Callable[[float], float] | None = None,
) -> float:

    y = func(x)
    below = y < y_crit
    if below.all():
        return x[-1] if origin == "right" else x[0]
    elif not below.any():
        return x[0] if origin == "left" else x[-1]

    indices = argwhere(below[:-1] ^ below[1:]).flatten()

    def _func(x):
        return func(x) - y_crit

    if indices.size > 1 and strategy == "average":
        idx_l = indices[0]
        left = newton(
            _func,
            x[idx_l] if origin == "left" else x[idx_l - 1],
            x1=x[idx_l + 1] if origin == "left" else x[idx_l],
            fprime=fprime,
        )

        idx_r = indices[-1]
        right = newton(
            _func,
            x[idx_r] if origin == "right" else x[idx_r + 1],
            x1=x[idx_r - 1] if origin == "right" else x[idx_r],
            fprime=fprime,
        )
        return (left + right) / 2

    if indices.size == 1 or strategy == "widest":
        idx = indices[0 if origin == "left" else -1]
    else:
        idx = indices[-1 if origin == "left" else 0]

    return newton(
        _func,
        x[idx] if origin == "left" else x[idx - 1],
        x1=x[idx + 1] if origin == "left" else x[idx],
        fprime=fprime,
    )


def get_width_at_height(
    model: Callable[[FloatVector], FloatVector]
    | GaussianSample
    | GaussianSampleList,
    x: FloatVector,
    x_max: float,
    y_crit: float,
    strategy: FWHMStrategy,
) -> float:
    is_left = x < x_max
    is_right = invert(is_left)

    if isinstance(model, (GaussianSample, GaussianSampleList)):
        func = model.__call__
        fprime = model.prime
    else:
        func = model
        fprime = None

    left_edge = x[0]
    if is_left.any():
        left_edge = get_intercept(
            func,
            x[is_left],
            y_crit,
            "left",
            strategy,
            fprime=fprime,
        )

    right_edge = x[-1]
    if is_right.any():
        right_edge = get_intercept(
            func,
            x[is_right],
            y_crit,
            "right",
            strategy,
            fprime=fprime,
        )

    return right_edge - left_edge


def get_quantile(
    q: float | Iterable[float],
    model: Callable[[FloatVector], FloatVector]
    | GaussianSample
    | GaussianSampleList,
    x: FloatVector,
) -> float | FloatVector:

    if isinstance(model, (GaussianSample, GaussianSampleList)):

        def func(a):
            nonlocal model, x
            return get_intercept(
                model.cdf, x, a, "left", "widest", fprime=model.pdf
            )
    else:

        def func(a):
            nonlocal model, x
            return get_intercept(model, x, a, "left", "widest")

    if isinstance(q, Iterable):
        return array([func(a) for a in q], dtype=float64)
    else:
        return func(q)


def get_peak(
    model: Callable[[FloatVector], FloatVector]
    | GaussianSample
    | GaussianSampleList,
    x: FloatVector,
) -> tuple[float, float]:
    """
    Finds the peak flux density of a model.

    Returns:
    --------
    x_max : float
        The x value at which the model reaches its maximum.
    y_max : float
        The maximum value of the model at x_max.
    """

    y = model(x)
    idx = y.argmax()
    if idx == 0:
        return x[0], y[0]
    elif idx == x.size - 1:
        return x[-1], y[-1]

    def func(x):
        return -model(x)

    res = minimize_scalar(func, bounds=(x[idx - 1], x[idx + 1]))
    x_max: float = res.x
    y_max: float = -res.fun

    return x_max, y_max


def get_median_absolute_deviation(
    cen: float,
    model: Callable[[FloatVector], FloatVector]
    | GaussianSample
    | GaussianSampleList,
    x: FloatVector,
) -> float:
    if isinstance(model, (GaussianSample, GaussianSampleList)):

        def func(x):
            nonlocal model
            return model.cdf(x + cen) - model.cdf(cen - x)

        def fprime(x):
            nonlocal model
            return model.pdf(x + cen) + model.pdf(cen - x)
    else:

        def func(x):
            nonlocal model
            return model(x + cen) - model(cen - x)

        fprime = None

    return get_intercept(func, x, 0.5, "left", "widest", fprime=fprime)


def get_average_absolute_deviation(
    cen: float,
    dists: rv_continuous_frozen | Iterable[rv_continuous_frozen],
    weights: FloatVector,
) -> float:
    if isinstance(dists, rv_continuous_frozen):
        mean, std = dists.args[:2]
        f0 = dists.pdf(cen)
        F0 = dists.cdf(cen)
        return (cen - mean) * (2 * F0 - 1) + 2 * std**2 * f0

    W = weights.sum()
    if W == 0:
        return 0.0

    aad_per_profile = [
        get_average_absolute_deviation(cen, dist, (1.0,)) for dist in dists
    ]

    return dot(weights, aad_per_profile) / W
