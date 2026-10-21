"""
Benchmark suite for @njit compilation time and performance speed-ups for measures/approx.py functions.

This test module measures:
1. Compilation time for @njit-decorated functions
2. Execution time for njit vs non-njit versions
3. Speed-up factors and break-even analysis
"""

import time
from typing import Callable
import pytest
import numpy as np
from numba import njit

# Import the njit-decorated versions from the actual module
from quasar_errors.measures.approx import (
    u_yinterp,
    u_xinterp,
    get_intercept,
    get_width_at_height,
    get_quantile,
    get_peak,
)


# ============================================================================
# Create non-njit versions of each function
# ============================================================================


def u_yinterp_no_njit(
    x: float,
    xp: np.ndarray,
    yp: np.ndarray,
    dyp: np.ndarray,
) -> tuple[float, float]:
    """Non-njit version of u_yinterp."""
    dx = xp[1] - xp[0]
    f0 = (xp[1] - x) / dx
    f1 = (x - xp[0]) / dx

    return (
        f0 * yp[0] + f1 * yp[1],
        np.hypot(f0 * dyp[0], f1 * dyp[1]),
    )


def u_xinterp_no_njit(
    y: float,
    xp: np.ndarray,
    yp: np.ndarray,
    dyp: np.ndarray,
) -> tuple[float, float]:
    """Non-njit version of u_xinterp."""
    dx = xp[1] - xp[0]
    dy = yp[1] - yp[0]
    f0 = (yp[1] - y) / dy
    f1 = (y - yp[0]) / dy

    return (
        f0 * xp[0] + f1 * xp[1],
        abs(dx) / abs(dy) * np.hypot(f0 * dyp[0], f1 * dyp[1]),
    )


def get_intercept_no_njit(
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    y_crit: float,
    strategy: str,
) -> tuple[float, float]:
    """Non-njit version of get_intercept."""
    below = y < y_crit
    if below.all():
        return (x[-1], dy[-1])
    elif not below.any():
        return (x[0], dy[0])

    above = np.invert(below)
    indices = np.argwhere(below[:-1] & above[1:]).flatten()

    if indices.size > 1 and strategy == "average":
        idx_r = indices[0]
        left = u_xinterp_no_njit(
            y_crit,
            x[idx_r : idx_r + 2],
            y[idx_r : idx_r + 2],
            dy[idx_r : idx_r + 2],
        )
        idx_l = indices[-1]
        right = u_xinterp_no_njit(
            y_crit,
            x[idx_l : idx_l + 2],
            y[idx_l : idx_l + 2],
            dy[idx_l : idx_l + 2],
        )

        return (
            (left[0] + right[0]) / 2,
            np.hypot(left[1], right[1]) / 2,
        )

    if indices.size == 1 or strategy == "widest":
        idx = indices[0]
    else:
        idx = indices[-1]

    return u_xinterp_no_njit(
        y_crit,
        x[idx : idx + 2],
        y[idx : idx + 2],
        dy[idx : idx + 2],
    )


def get_width_at_height_no_njit(
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    x_max: float,
    y_crit: float,
    strategy: str,
) -> tuple[float, float]:
    """Non-njit version of get_width_at_height."""
    is_left = x < x_max
    is_right = np.invert(is_left)

    left_edge = (x[0], 0)
    if is_left.any():
        left_edge = get_intercept_no_njit(
            x[is_left],
            y[is_left],
            dy[is_left],
            y_crit,
            strategy=strategy,
        )

    right_edge = (x[-1], 0)
    if is_right.any():
        right_edge = get_intercept_no_njit(
            x[is_right][::-1],
            y[is_right][::-1],
            dy[is_right][::-1],
            y_crit,
            strategy=strategy,
        )

    return (
        right_edge[0] - left_edge[0],
        np.hypot(left_edge[1], right_edge[1]),
    )


def get_quantile_no_njit(
    q: float | np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float | np.ndarray,
) -> float | np.ndarray:
    """Non-njit version of get_quantile."""
    edges = np.append(
        x - dx / 2,
        x[-1] + (dx if isinstance(dx, float) else dx[-1]) / 2,
    )
    cdf = np.append(0, y.cumsum())
    cdf /= cdf[-1]
    return np.interp(q, cdf, edges)


def get_peak_no_njit(
    x: np.ndarray,
    y: np.ndarray,
    dx: float | np.ndarray,
) -> tuple[float, float]:
    """Non-njit version of get_peak."""
    edges = np.append(
        x - dx / 2,
        x[-1] + (dx if isinstance(dx, float) else dx[-1]) / 2,
    )
    idx = y.argmax()
    if isinstance(idx, np.ndarray):
        idx = idx.flatten()[0]
    y_max = y[idx]

    if idx == 0 or idx == x.size - 1:
        x_max = x[idx]
    else:
        x_l, x_r = edges[idx : idx + 2]
        y_l, _, y_r = y[idx - 1 : idx + 2]
        x_max = x_l + (x_r - x_l) * ((y_max - y_l) / (2 * y_max - y_l - y_r))

    return (x_max, y_max)


# ============================================================================
# Benchmark infrastructure
# ============================================================================


def benchmark_function(
    func: Callable,
    test_args: dict,
    iterations: int = 100,
) -> float:
    """
    Benchmark a function's execution time.

    Args:
        func: Function to benchmark
        test_args: Arguments to pass
        iterations: Number of iterations to run

    Returns:
        Median execution time per call in milliseconds
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(**test_args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times.sort()
    return np.median(times) * 1000  # Convert to ms


def calculate_speedup(time_no_njit: float, time_njit: float) -> float:
    """Calculate speedup factor."""
    if time_njit == 0:
        return float("inf")
    return time_no_njit / time_njit


def calculate_breakeven(
    compilation_ms: float, time_no_njit: float, time_njit: float
) -> int:
    """Calculate break-even point (number of calls)."""
    time_saved_per_call = time_no_njit - time_njit
    if time_saved_per_call <= 0:
        return float("inf")
    return int(np.ceil(compilation_ms / time_saved_per_call))


# ============================================================================
# Test data generation
# ============================================================================


def generate_interpolation_data(size: int = 100) -> dict:
    """Generate data for interpolation functions."""
    x = np.linspace(0, 10, size)
    y = np.sin(x) + 0.1 * np.random.randn(size)
    dy = np.abs(0.01 * np.random.randn(size))
    return {
        "x": x,
        "y": y,
        "dy": dy,
    }


def generate_simple_interp_data() -> dict:
    """Generate 2-point data for simple interpolation."""
    xp = np.array([0.0, 1.0])
    yp = np.array([0.5, 1.5])
    dyp = np.array([0.05, 0.05])
    return {
        "xp": xp,
        "yp": yp,
        "dyp": dyp,
        "x": 0.5,
        "y": 1.0,
    }


def generate_get_intercept_data(size: int = 100) -> dict:
    """Generate data for get_intercept."""
    data = generate_interpolation_data(size)
    return {
        "x": data["x"],
        "y": data["y"],
        "dy": data["dy"],
        "y_crit": 0.5,
        "strategy": "widest",
    }


def generate_get_width_data(size: int = 100) -> dict:
    """Generate data for get_width_at_height."""
    data = generate_interpolation_data(size)
    return {
        "x": data["x"],
        "y": data["y"],
        "dy": data["dy"],
        "x_max": 5.0,
        "y_crit": 0.5,
        "strategy": "widest",
    }


def generate_get_quantile_data(size: int = 100) -> dict:
    """Generate data for get_quantile."""
    x = np.linspace(0, 10, size)
    y = np.exp(-x)
    dx = x[1] - x[0]
    return {
        "q": 0.5,
        "x": x,
        "y": y / y.sum(),
        "dx": dx,
    }


def generate_get_peak_data(size: int = 100) -> dict:
    """Generate data for get_peak."""
    x = np.linspace(-5, 5, size)
    y = np.exp(-(x**2 / 2))
    dx = x[1] - x[0]
    return {
        "x": x,
        "y": y,
        "dx": dx,
    }


# ============================================================================
# Compilation time tests
# ============================================================================


def test_u_yinterp_compilation_time():
    """Test compilation time for u_yinterp."""
    test_data = generate_simple_interp_data()
    test_data.pop("y")  # Remove 'y' key, not needed for u_yinterp

    start = time.perf_counter()
    u_yinterp(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0
    print(f"\nu_yinterp compilation time: {compilation_time * 1000:.2f} ms")


def test_u_xinterp_compilation_time():
    """Test compilation time for u_xinterp."""
    test_data = generate_simple_interp_data()
    test_data.pop("x")  # Remove 'x' key, not needed for u_xinterp

    start = time.perf_counter()
    u_xinterp(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0
    print(f"\nu_xinterp compilation time: {compilation_time * 1000:.2f} ms")


def test_get_intercept_compilation_time():
    """Test compilation time for get_intercept."""
    test_data = generate_get_intercept_data()

    start = time.perf_counter()
    get_intercept(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0
    print(
        f"\nget_intercept compilation time: {compilation_time * 1000:.2f} ms"
    )


def test_get_width_at_height_compilation_time():
    """Test compilation time for get_width_at_height."""
    test_data = generate_get_width_data()

    start = time.perf_counter()
    get_width_at_height(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0
    print(
        f"\nget_width_at_height compilation time: {compilation_time * 1000:.2f} ms"
    )


def test_get_quantile_compilation_time():
    """Test compilation time for get_quantile."""
    test_data = generate_get_quantile_data()

    start = time.perf_counter()
    get_quantile(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0
    print(f"\nget_quantile compilation time: {compilation_time * 1000:.2f} ms")


# NOTE: get_peak has numba compilation issues with argmax().flatten() pattern
# and is skipped from benchmarking


# ============================================================================
# Speed-up tests
# ============================================================================


def test_u_yinterp_speedup():
    """Test speed-up for u_yinterp."""
    test_data = generate_simple_interp_data()
    test_data.pop("y")

    u_yinterp(**test_data)  # Warm up

    time_no_njit = benchmark_function(
        u_yinterp_no_njit, {**test_data, "x": test_data["x"]}, iterations=1000
    )
    time_njit = benchmark_function(u_yinterp, test_data, iterations=1000)
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(
        f"\nu_yinterp speedup ({speedup:.2f}x): No-NJIT {time_no_njit:.6f}ms vs NJIT {time_njit:.6f}ms"
    )


def test_u_xinterp_speedup():
    """Test speed-up for u_xinterp."""
    test_data = generate_simple_interp_data()
    test_data.pop("x")

    u_xinterp(**test_data)  # Warm up

    time_no_njit = benchmark_function(
        u_xinterp_no_njit, {**test_data, "y": test_data["y"]}, iterations=1000
    )
    time_njit = benchmark_function(u_xinterp, test_data, iterations=1000)
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(
        f"\nu_xinterp speedup ({speedup:.2f}x): No-NJIT {time_no_njit:.6f}ms vs NJIT {time_njit:.6f}ms"
    )


def test_get_intercept_speedup():
    """Test speed-up for get_intercept."""
    test_data = generate_get_intercept_data()

    get_intercept(**test_data)  # Warm up

    time_no_njit = benchmark_function(
        get_intercept_no_njit, test_data, iterations=100
    )
    time_njit = benchmark_function(get_intercept, test_data, iterations=100)
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(
        f"\nget_intercept speedup ({speedup:.2f}x): No-NJIT {time_no_njit:.4f}ms vs NJIT {time_njit:.4f}ms"
    )


def test_get_width_at_height_speedup():
    """Test speed-up for get_width_at_height."""
    test_data = generate_get_width_data()

    get_width_at_height(**test_data)  # Warm up

    time_no_njit = benchmark_function(
        get_width_at_height_no_njit, test_data, iterations=100
    )
    time_njit = benchmark_function(
        get_width_at_height, test_data, iterations=100
    )
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(
        f"\nget_width_at_height speedup ({speedup:.2f}x): No-NJIT {time_no_njit:.4f}ms vs NJIT {time_njit:.4f}ms"
    )


def test_get_quantile_speedup():
    """Test speed-up for get_quantile."""
    test_data = generate_get_quantile_data()

    get_quantile(**test_data)  # Warm up

    time_no_njit = benchmark_function(
        get_quantile_no_njit, test_data, iterations=100
    )
    time_njit = benchmark_function(get_quantile, test_data, iterations=100)
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(
        f"\nget_quantile speedup ({speedup:.2f}x): No-NJIT {time_no_njit:.4f}ms vs NJIT {time_njit:.4f}ms"
    )


# NOTE: get_peak has numba compilation issues and speedup test is skipped


# ============================================================================
# Comprehensive summary benchmark
# ============================================================================


def test_approx_functions_summary():
    """Run comprehensive benchmark of all approx.py functions and print summary."""

    print("\n" + "=" * 100)
    print("COMPREHENSIVE NJIT PERFORMANCE BENCHMARK - measures/approx.py")
    print("=" * 100)

    # Compilation times (measured separately with first calls)
    compilation_times = {}

    print("\n📦 COMPILATION TIMES:")
    print("-" * 100)

    # u_yinterp
    test_data = generate_simple_interp_data()
    test_data.pop("y")
    start = time.perf_counter()
    u_yinterp(**test_data)
    compilation_times["u_yinterp"] = (time.perf_counter() - start) * 1000
    print(
        f"u_yinterp:              {compilation_times['u_yinterp']:>10.2f} ms"
    )

    # u_xinterp
    test_data = generate_simple_interp_data()
    test_data.pop("x")
    start = time.perf_counter()
    u_xinterp(**test_data)
    compilation_times["u_xinterp"] = (time.perf_counter() - start) * 1000
    print(
        f"u_xinterp:              {compilation_times['u_xinterp']:>10.2f} ms"
    )

    # get_intercept
    test_data = generate_get_intercept_data()
    start = time.perf_counter()
    get_intercept(**test_data)
    compilation_times["get_intercept"] = (time.perf_counter() - start) * 1000
    print(
        f"get_intercept:          {compilation_times['get_intercept']:>10.2f} ms"
    )

    # get_width_at_height
    test_data = generate_get_width_data()
    start = time.perf_counter()
    get_width_at_height(**test_data)
    compilation_times["get_width_at_height"] = (
        time.perf_counter() - start
    ) * 1000
    print(
        f"get_width_at_height:    {compilation_times['get_width_at_height']:>10.2f} ms"
    )

    # get_quantile
    test_data = generate_get_quantile_data()
    start = time.perf_counter()
    get_quantile(**test_data)
    compilation_times["get_quantile"] = (time.perf_counter() - start) * 1000
    print(
        f"get_quantile:           {compilation_times['get_quantile']:>10.2f} ms"
    )

    # get_peak - skip for now due to numba compilation issues
    print(f"get_peak:               (skipped - numba compilation issue)")

    # Speed-up benchmarks
    print("\n⚡ PERFORMANCE BENCHMARKS (100-element arrays):")
    print("-" * 100)
    print(
        f"{'Function':<30} {'No-NJIT (ms)':>18} {'NJIT (ms)':>18} {'Speedup':>15} {'Break-even':>15}"
    )
    print("-" * 100)

    results = {}

    # u_yinterp
    test_data = generate_simple_interp_data()
    test_data.pop("y")
    time_no_njit = benchmark_function(
        u_yinterp_no_njit, {**test_data, "x": test_data["x"]}, iterations=500
    )
    time_njit = benchmark_function(u_yinterp, test_data, iterations=500)
    speedup = calculate_speedup(time_no_njit, time_njit)
    breakeven = calculate_breakeven(
        compilation_times["u_yinterp"], time_no_njit, time_njit
    )
    results["u_yinterp"] = (time_no_njit, time_njit, speedup, breakeven)
    print(
        f"{'u_yinterp':<30} {time_no_njit:>18.6f} {time_njit:>18.6f} {speedup:>15.2f}x {breakeven:>15,.0f}"
    )

    # u_xinterp
    test_data = generate_simple_interp_data()
    test_data.pop("x")
    time_no_njit = benchmark_function(
        u_xinterp_no_njit, {**test_data, "y": test_data["y"]}, iterations=500
    )
    time_njit = benchmark_function(u_xinterp, test_data, iterations=500)
    speedup = calculate_speedup(time_no_njit, time_njit)
    breakeven = calculate_breakeven(
        compilation_times["u_xinterp"], time_no_njit, time_njit
    )
    results["u_xinterp"] = (time_no_njit, time_njit, speedup, breakeven)
    print(
        f"{'u_xinterp':<30} {time_no_njit:>18.6f} {time_njit:>18.6f} {speedup:>15.2f}x {breakeven:>15,.0f}"
    )

    # get_intercept
    test_data = generate_get_intercept_data()
    time_no_njit = benchmark_function(
        get_intercept_no_njit, test_data, iterations=50
    )
    time_njit = benchmark_function(get_intercept, test_data, iterations=50)
    speedup = calculate_speedup(time_no_njit, time_njit)
    breakeven = calculate_breakeven(
        compilation_times["get_intercept"], time_no_njit, time_njit
    )
    results["get_intercept"] = (time_no_njit, time_njit, speedup, breakeven)
    print(
        f"{'get_intercept':<30} {time_no_njit:>18.4f} {time_njit:>18.4f} {speedup:>15.2f}x {breakeven:>15,.0f}"
    )

    # get_width_at_height
    test_data = generate_get_width_data()
    time_no_njit = benchmark_function(
        get_width_at_height_no_njit, test_data, iterations=50
    )
    time_njit = benchmark_function(
        get_width_at_height, test_data, iterations=50
    )
    speedup = calculate_speedup(time_no_njit, time_njit)
    breakeven = calculate_breakeven(
        compilation_times["get_width_at_height"], time_no_njit, time_njit
    )
    results["get_width_at_height"] = (
        time_no_njit,
        time_njit,
        speedup,
        breakeven,
    )
    print(
        f"{'get_width_at_height':<30} {time_no_njit:>18.4f} {time_njit:>18.4f} {speedup:>15.2f}x {breakeven:>15,.0f}"
    )

    # get_quantile
    test_data = generate_get_quantile_data()
    time_no_njit = benchmark_function(
        get_quantile_no_njit, test_data, iterations=50
    )
    time_njit = benchmark_function(get_quantile, test_data, iterations=50)
    speedup = calculate_speedup(time_no_njit, time_njit)
    breakeven = calculate_breakeven(
        compilation_times["get_quantile"], time_no_njit, time_njit
    )
    results["get_quantile"] = (time_no_njit, time_njit, speedup, breakeven)
    print(
        f"{'get_quantile':<30} {time_no_njit:>18.4f} {time_njit:>18.4f} {speedup:>15.2f}x {breakeven:>15,.0f}"
    )

    # Summary statistics
    avg_speedup = np.mean([r[2] for r in results.values()])
    avg_breakeven = np.mean([r[3] for r in results.values()])

    print("-" * 100)
    print(
        f"{'AVERAGE':<30} {' ' * 18} {' ' * 18} {avg_speedup:>15.2f}x {avg_breakeven:>15,.0f}"
    )
    print("=" * 100)
