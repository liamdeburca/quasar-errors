import numpy as np
from uncertainties import ufloat

from quasar_errors.measures.approx import u_xinterp, u_yinterp


def u_interp(
    x,
    xp: tuple[float, float],
    yp: tuple[float, float],
    *,
    dxp: tuple[float, float] = (0, 0),
    dyp: tuple[float, float] = (0, 0),
) -> tuple[float, float]:
    x0 = ufloat(xp[0], dxp[0])
    x1 = ufloat(xp[1], dxp[1])

    y0 = ufloat(yp[0], dyp[0])
    y1 = ufloat(yp[1], dyp[1])

    f0 = (x1 - x) / (x1 - x0)
    f1 = (x - x0) / (x1 - x0)

    y = f0 * y0 + f1 * y1

    return (
        y.nominal_value,
        y.std_dev,
    )


def test_u_yinterp_100_random_points():
    """
    Test u_yinterp against uncertainties package on 100 random points.

    Both use 1st order Taylor expansion for error propagation, so results
    should be nearly identical with very small tolerance.
    """
    np.random.seed(42)

    # Generate random signal
    n = 50
    x_vals = np.linspace(0, 10, n)
    y_vals = np.sin(x_vals) + np.random.randn(n) * 0.05
    dy_vals = 0.01 + np.random.rand(n) * 0.05

    # Test on 100 random interpolation points
    for test_num in range(100):
        idx = np.random.randint(0, n - 1)
        xp = np.array([x_vals[idx], x_vals[idx + 1]], dtype=np.float64)
        yp = np.array([y_vals[idx], y_vals[idx + 1]], dtype=np.float64)
        dyp = np.array([dy_vals[idx], dy_vals[idx + 1]], dtype=np.float64)

        x_interp = xp[0] + np.random.rand() * (xp[1] - xp[0])

        y_result, dy_result = u_yinterp(x_interp, xp, yp, dyp)
        y_ref, dy_red = u_interp(x_interp, xp, yp, dyp=dyp)

        assert np.isclose(y_result, y_ref, rtol=1e-12, atol=1e-14), (
            f"Test {test_num}: y nominal value mismatch"
        )
        assert np.isclose(dy_result, dy_red, rtol=1e-12, atol=1e-14), (
            f"Test {test_num}: y error mismatch"
        )


def test_u_xinterp_100_random_points():
    """
    Test u_xinterp against uncertainties package on 100 random points.

    Both use 1st order Taylor expansion for error propagation, so results
    should be nearly identical with very small tolerance.
    """
    np.random.seed(42)

    # Generate random signal
    n = 50
    x_vals = np.linspace(0, 10, n)
    y_vals = np.sin(x_vals) + np.random.randn(n) * 0.05
    dy_vals = 0.01 + np.random.rand(n) * 0.05

    # Test on 100 random interpolation points
    for test_num in range(100):
        # Pick two adjacent reference points
        idx = np.random.randint(0, n - 1)
        xp = np.array([x_vals[idx], x_vals[idx + 1]], dtype=np.float64)
        yp = np.array([y_vals[idx], y_vals[idx + 1]], dtype=np.float64)
        dyp = np.array([dy_vals[idx], dy_vals[idx + 1]], dtype=np.float64)

        y_interp = yp[0] + np.random.rand() * (yp[1] - yp[0])

        x_result, dx_result = u_xinterp(y_interp, xp, yp, dyp)
        x_ref, dx_ref = u_interp(y_interp, yp, xp, dxp=dyp)

        assert np.isclose(x_result, x_ref, rtol=1e-12, atol=1e-14), (
            f"Test {test_num}: x nominal value mismatch"
        )
        assert np.isclose(dx_result, dx_ref, rtol=1e-12, atol=1e-14), (
            f"Test {test_num}: x error mismatch"
        )
