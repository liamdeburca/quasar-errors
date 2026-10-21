"""
Python version of get_intercept_c Cython function using numpy and uncertainties.
"""
import numpy as np
import pytest

from quasar_errors.measures.approx import get_intercept


def u_xinterp(
    y: float,
    xp: np.ndarray,
    yp: np.ndarray,
    dyp: np.ndarray,
) -> tuple[float, float]:
    dx = xp[1] - xp[0]
    dy = yp[1] - yp[0]
    
    f0 = (yp[1] - y) / dy
    f1 = (y - yp[0]) / dy
    
    dy_term0 = f0 * dyp[0]
    dy_term1 = f1 * dyp[1]
    scale = np.abs(dx) / np.abs(dy)
    
    x_interp = f0 * xp[0] + f1 * xp[1]
    u_x = scale * np.sqrt(dy_term0**2 + dy_term1**2)
    
    return (x_interp, u_x)


def get_intercept_py(
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    y_crit: float,
    strategy: int = 0,
) -> tuple[float, float]:
    n = len(x)
    first_crossing = -1
    last_crossing = -1
    n_crossings = 0
    
    # Find all UPWARD crossings (y[i] < y_crit and y[i+1] >= y_crit)
    for i in range(n - 1):
        if (y[i] < y_crit) and (y[i + 1] >= y_crit):
            if first_crossing == -1:
                first_crossing = i
            last_crossing = i
            n_crossings += 1
    
    # Handle edge cases
    if n_crossings == 0:
        if y[0] >= y_crit:
            return (float(x[0]), float(dy[0]))
        else:
            return (float(x[n - 1]), float(dy[n - 1]))
    
    # Single crossing: use it
    if n_crossings == 1:
        idx_r = first_crossing
        return u_xinterp(
            y_crit,
            x[idx_r : idx_r + 2],
            y[idx_r : idx_r + 2],
            dy[idx_r : idx_r + 2],
        )
    
    # Multiple crossings: apply strategy
    if strategy == 1:  # average
        # First crossing
        idx_r = first_crossing
        left_x, left_dy = u_xinterp(
            y_crit,
            x[idx_r : idx_r + 2],
            y[idx_r : idx_r + 2],
            dy[idx_r : idx_r + 2],
        )
        
        # Last crossing
        idx_r = last_crossing
        right_x, right_dy = u_xinterp(
            y_crit,
            x[idx_r : idx_r + 2],
            y[idx_r : idx_r + 2],
            dy[idx_r : idx_r + 2],
        )
        
        avg_x = (left_x + right_x) / 2.0
        avg_dy = np.sqrt(left_dy**2 + right_dy**2) / 2.0
        return (avg_x, avg_dy)
    elif strategy == 2:  # widest (first crossing)
        idx_r = first_crossing
    else:  # narrowest (last crossing)
        idx_r = last_crossing
    
    return u_xinterp(
        y_crit,
        x[idx_r : idx_r + 2],
        y[idx_r : idx_r + 2],
        dy[idx_r : idx_r + 2],
    )

def test_no_noise(subtests: pytest.Subtests):
    N = 100
    rng = np.random.default_rng(42)

    x = np.linspace(-5, 5, N)
    dy = np.full_like(x, 0.1)
    y = rng.normal(loc=2 * x, scale=0)

    for i in range(100):
        y_crit = rng.uniform(low=y.min(), high=y.max())
        with subtests.test(i):
            control = y_crit / 2
            test_py = get_intercept_py(x, y, dy, y_crit, 1)
            test_c = get_intercept(x, y, dy, y_crit, "average")

            assert np.isclose(control, test_py[0])
            assert np.isclose(control, test_c[0])

def test_noisy(subtests: pytest.Subtests):
    N = 100
    rng = np.random.default_rng(42)

    x = np.linspace(-5, 5, N)
    dy = np.full_like(x, 0.1)

    for i in range(100):
        y = rng.normal(loc=2 * x, scale=dy)
        y_crit = rng.uniform(low=y.min(), high=y.max())
        with subtests.test(i):
            control = get_intercept_py(x, y, dy, y_crit, 1)
            test = get_intercept(x, y, dy, y_crit, "average")

            assert np.isclose(control[0], test[0])
            assert np.isclose(control[1], test[1])