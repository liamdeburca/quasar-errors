import time
import numpy as np
from typing import Tuple
from uncertainties import ufloat
from itertools import starmap

from src.quasar_errors.measures.approx import u_yinterp

def u_yinterp_ref(
    x,
    xp: Tuple[float, float],
    yp: Tuple[float, float],
    dyp: Tuple[float, float],
) -> Tuple[float, float]:
    """Reference implementation using uncertainties package."""
    x0, x1 = xp

    y0 = ufloat(yp[0], dyp[0])
    y1 = ufloat(yp[1], dyp[1])

    f0 = (x1 - x) / (x1 - x0)
    f1 = (x - x0) / (x1 - x0)

    y = f0 * y0 + f1 * y1

    return (
        y.nominal_value,
        y.std_dev,
    )


if __name__ == "__main__":
    np.random.seed(42)
    
    # Setup
    n = 50
    x_vals = np.linspace(0, 10, n)
    y_vals = np.sin(x_vals) + np.random.randn(n) * 0.05
    dy_vals = 0.01 + np.random.rand(n) * 0.05
    
    # Generate test data: 10,000 interpolations
    test_cases = []
    for _ in range(10000):
        idx = np.random.randint(0, n - 1)
        xp = (x_vals[idx], x_vals[idx + 1])
        yp = (y_vals[idx], y_vals[idx + 1])
        dyp = (dy_vals[idx], dy_vals[idx + 1])
        x_interp = xp[0] + np.random.rand() * (xp[1] - xp[0])
        test_cases.append((x_interp, xp, yp, dyp))
    
    # Benchmark u_yinterp
    _ = u_yinterp(*test_cases[0])  # Warm up JIT

    start = time.perf_counter()
    _ = list(starmap(u_yinterp, test_cases))
    t_u_yinterp = time.perf_counter() - start
    
    # Benchmark u_yinterp_ref
    start = time.perf_counter()
    _ = list(starmap(u_yinterp_ref, test_cases))
    t_u_yinterp_ref = time.perf_counter() - start
    
    speedup = t_u_yinterp_ref / t_u_yinterp
    
    print(f"u_yinterp:      {t_u_yinterp:.4f} s")
    print(f"u_yinterp_ref: {t_u_yinterp_ref:.4f} s")
    print(f"\nSpeedup: {speedup:.1f}x faster")
    print(f"Per-call improvement: {t_u_yinterp_ref / t_u_yinterp:.0%}")
