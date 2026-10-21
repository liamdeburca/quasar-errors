import numpy as np
import pytest

from quasar_errors.measures.approx import (
    get_intercept,
    get_left_edge_at_height,
    get_right_edge_at_height,
)


def get_left_edge_py(
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    mode: float,
    y_crit: float,
) -> tuple[float, float]:
    mode_idx = np.searchsorted(x, mode, side="left")
    
    val = x[0]
    unc = 0.0
    if mode_idx > 0:
        val, unc = get_intercept(
            x[:mode_idx],
            y[:mode_idx] - y_crit,
            dy[:mode_idx],
            0.0,
            "average",
        )

    return val, unc

def get_right_edge_py(
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    mode: float,
    y_crit: float,
) -> tuple[float, float]:
    mode_idx = np.searchsorted(x, mode, side="right") - 1
    
    val = x[-1]
    unc = 0.0
    if mode_idx < len(x):
        val, unc = get_intercept(
            x[mode_idx:],
            y_crit - y[mode_idx:],
            dy[mode_idx:],
            0.0,
            "average",
        )

    return val, unc


def test_no_noise(subtests: pytest.Subtests):
    N = 100
    rng = np.random.default_rng(42)

    x = np.linspace(-1, 1, 100)
    y = 1 - x ** 2
    dy = np.full_like(x, 0.1)

    isclose = lambda a, b: np.isclose(a, b, atol=1e-3)

    for i in range(N):
        y_crit = rng.uniform(low=y.min(), high=y.max())
        with subtests.test(f"{i=}, {y_crit=:.2f},"):
            control_right = (1 - y_crit) ** 0.5
            control_left = -control_right

            test_py = get_left_edge_py(x, y, dy, 0.0, y_crit)
            assert isclose(control_left, test_py[0])
            test_c = get_left_edge_at_height(x, y, dy, 0.0, y_crit, "average")
            assert isclose(control_left, test_c[0])

            test_py = get_right_edge_py(x, y, dy, 0.0, y_crit)
            assert isclose(control_right, test_py[0])
            test_c = get_right_edge_at_height(x, y, dy, 0.0, y_crit, "average")
            assert isclose(control_right, test_c[0])

