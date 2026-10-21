"""
Benchmark suite for @njit compilation time and performance speed-ups.

This test module measures:
1. Compilation time for @njit-decorated functions
2. Execution time for njit vs non-njit versions
3. Speed-up factors across different input sizes
"""

import time
from typing import Callable
import pytest
import numpy as np
from numba import njit

# Import the non-njit versions from the actual module
from quasar_errors.analysis.inner import (
    second_moment,
    weighted_second_moment,
    third_moment,
    weighted_third_moment,
    fourth_moment,
    weighted_fourth_moment,
)


# ============================================================================
# Create @njit-decorated versions of each function
# ============================================================================


@njit(fastmath=True)
def second_moment_njit(
    first_moment: float | np.ndarray,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    return sigma**2 + (mu - first_moment) ** 2


@njit(fastmath=True)
def weighted_second_moment_njit(
    weights: np.ndarray,
    first_moment: float | np.ndarray,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    return np.dot(weights, sigma**2 + (mu - first_moment) ** 2)


@njit(fastmath=True)
def third_moment_njit(
    first_moment: float | np.ndarray,
    second_moment: float | np.ndarray,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    result = 3 * mu * sigma**2
    result += mu**3
    result -= 2 * first_moment * (mu**2 + sigma**2)
    result += mu * first_moment**2
    result -= first_moment * second_moment
    return result


@njit(fastmath=True)
def weighted_third_moment_njit(
    weights: np.ndarray,
    first_moment: float | np.ndarray,
    second_moment: float | np.ndarray,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    return np.dot(
        weights, third_moment_njit(first_moment, second_moment, mu, sigma)
    )


@njit(fastmath=True)
def fourth_moment_njit(
    first_moment: float | np.ndarray,
    third_moment: float | np.ndarray,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    mu_sq = mu**2
    sigma_sq = sigma**2

    result = mu_sq**2 + 6 * mu_sq * sigma_sq + 3 * sigma_sq**2
    result -= 3 * first_moment * mu * (mu_sq + 3 * sigma_sq)
    result += 3 * first_moment**2 * (mu_sq + sigma_sq)
    result -= first_moment**3 * mu
    result -= first_moment * third_moment

    return result


@njit(fastmath=True)
def weighted_fourth_moment_njit(
    weights: np.ndarray,
    first_moment: float | np.ndarray,
    third_moment: float | np.ndarray,
    mu: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    return np.dot(
        weights, fourth_moment_njit(first_moment, third_moment, mu, sigma)
    )


# ============================================================================
# Compilation timing and benchmark infrastructure
# ============================================================================


class CompilationTimer:
    """Context manager to measure JIT compilation time."""

    def __init__(self):
        self.compilation_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.compilation_time = time.perf_counter() - self.start_time


def measure_compilation_time(njit_func: Callable, test_args: dict) -> float:
    """
    Measure the compilation time for an @njit function by calling it for the first time.

    Args:
        njit_func: The @njit-decorated function
        test_args: Dictionary of arguments to pass to the function

    Returns:
        Compilation time in seconds
    """
    # Clear any existing cached compilations
    if hasattr(njit_func, "_compilation_cached"):
        # For fresh compilations, we time the first call
        pass

    with CompilationTimer() as timer:
        njit_func(**test_args)

    return timer.compilation_time


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
        Average execution time per call in milliseconds
    """
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(**test_args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    # Use median to avoid outliers
    times.sort()
    return np.median(times) * 1000  # Convert to ms


def calculate_speedup(time_no_njit: float, time_njit: float) -> float:
    """Calculate speedup factor."""
    if time_njit == 0:
        return float("inf")
    return time_no_njit / time_njit


# ============================================================================
# Test fixtures for various data sizes
# ============================================================================


@pytest.fixture(
    params=[
        {"name": "scalar", "size": None},
        {"name": "small_array", "size": 10},
        {"name": "medium_array", "size": 1000},
        {"name": "large_array", "size": 100000},
    ]
)
def test_data_sizes(request):
    """Parametrized fixture providing different data sizes."""
    return request.param


def generate_test_data(size: int | None) -> dict:
    """Generate random test data for the given size."""
    if size is None:
        # Scalar test
        return {
            "first_moment": 5.0,
            "mu": 4.8,
            "sigma": 0.5,
        }
    else:
        # Array test
        return {
            "first_moment": np.random.randn(size),
            "mu": np.random.randn(size),
            "sigma": np.abs(np.random.randn(size)),
        }


def generate_weighted_test_data(size: int | None) -> dict:
    """Generate test data for weighted functions."""
    base_data = generate_test_data(size)
    if size is None:
        base_data["weights"] = np.array([0.3, 0.7])
    else:
        base_data["weights"] = np.abs(np.random.randn(size))
        base_data["weights"] /= base_data["weights"].sum()
    return base_data


def generate_moment_test_data(size: int | None, n: int) -> dict:
    """Generate test data for nth moment functions.

    Note: third_moment needs second_moment, fourth_moment needs third_moment (not second_moment)
    """
    base_data = generate_test_data(size)
    if n == 2:
        base_data["second_moment"] = np.abs(
            np.random.randn()
            if size is None
            else np.abs(np.random.randn(size))
        )
    elif n == 3:
        base_data["third_moment"] = (
            np.random.randn() if size is None else np.random.randn(size)
        )
    elif n >= 4:
        base_data["third_moment"] = (
            np.random.randn() if size is None else np.random.randn(size)
        )
    return base_data


# ============================================================================
# Tests for second_moment
# ============================================================================


def test_second_moment_compilation_time():
    """Test compilation time for second_moment."""
    test_data = generate_test_data(1000)

    start = time.perf_counter()
    second_moment_njit(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0, "Compilation time should be measurable"
    print(
        f"\nsecond_moment_njit compilation time: {compilation_time * 1000:.2f} ms"
    )


def test_second_moment_speedup(test_data_sizes):
    """Test speed-up factor for second_moment across different sizes."""
    size = test_data_sizes["size"]
    label = test_data_sizes["name"]

    test_data = generate_test_data(size)

    # Warm up njit version
    second_moment_njit(**test_data)

    # Benchmark
    time_no_njit = benchmark_function(second_moment, test_data, iterations=100)
    time_njit = benchmark_function(
        second_moment_njit, test_data, iterations=100
    )
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(f"\nsecond_moment ({label}):")
    print(f"  Non-njit: {time_no_njit:.4f} ms")
    print(f"  Njit:     {time_njit:.4f} ms")
    print(f"  Speedup:  {speedup:.2f}x")


# ============================================================================
# Tests for weighted_second_moment
# ============================================================================


def test_weighted_second_moment_compilation_time():
    """Test compilation time for weighted_second_moment."""
    test_data = generate_weighted_test_data(1000)

    start = time.perf_counter()
    weighted_second_moment_njit(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0, "Compilation time should be measurable"
    print(
        f"\nweighted_second_moment_njit compilation time: {compilation_time * 1000:.2f} ms"
    )


def test_weighted_second_moment_speedup(test_data_sizes):
    """Test speed-up factor for weighted_second_moment across different sizes."""
    size = test_data_sizes["size"]
    label = test_data_sizes["name"]

    if size is None:
        pytest.skip("Weighted functions need array inputs")

    test_data = generate_weighted_test_data(size)

    # Warm up njit version
    weighted_second_moment_njit(**test_data)

    # Benchmark
    time_no_njit = benchmark_function(
        weighted_second_moment, test_data, iterations=100
    )
    time_njit = benchmark_function(
        weighted_second_moment_njit, test_data, iterations=100
    )
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(f"\nweighted_second_moment ({label}):")
    print(f"  Non-njit: {time_no_njit:.4f} ms")
    print(f"  Njit:     {time_njit:.4f} ms")
    print(f"  Speedup:  {speedup:.2f}x")


# ============================================================================
# Tests for third_moment
# ============================================================================


def test_third_moment_compilation_time():
    """Test compilation time for third_moment."""
    test_data = generate_moment_test_data(1000, n=2)

    start = time.perf_counter()
    third_moment_njit(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0, "Compilation time should be measurable"
    print(
        f"\nthird_moment_njit compilation time: {compilation_time * 1000:.2f} ms"
    )


def test_third_moment_speedup(test_data_sizes):
    """Test speed-up factor for third_moment across different sizes."""
    size = test_data_sizes["size"]
    label = test_data_sizes["name"]

    test_data = generate_moment_test_data(size, n=2)

    # Warm up njit version
    third_moment_njit(**test_data)

    # Benchmark
    time_no_njit = benchmark_function(third_moment, test_data, iterations=100)
    time_njit = benchmark_function(
        third_moment_njit, test_data, iterations=100
    )
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(f"\nthird_moment ({label}):")
    print(f"  Non-njit: {time_no_njit:.4f} ms")
    print(f"  Njit:     {time_njit:.4f} ms")
    print(f"  Speedup:  {speedup:.2f}x")


# ============================================================================
# Tests for weighted_third_moment
# ============================================================================


def test_weighted_third_moment_compilation_time():
    """Test compilation time for weighted_third_moment."""
    test_data = generate_weighted_test_data(1000)
    test_data["second_moment"] = np.abs(np.random.randn(1000))

    start = time.perf_counter()
    weighted_third_moment_njit(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0, "Compilation time should be measurable"
    print(
        f"\nweighted_third_moment_njit compilation time: {compilation_time * 1000:.2f} ms"
    )


def test_weighted_third_moment_speedup(test_data_sizes):
    """Test speed-up factor for weighted_third_moment across different sizes."""
    size = test_data_sizes["size"]
    label = test_data_sizes["name"]

    if size is None:
        pytest.skip("Weighted functions need array inputs")

    test_data = generate_weighted_test_data(size)
    test_data["second_moment"] = np.abs(np.random.randn(size))

    # Warm up njit version
    weighted_third_moment_njit(**test_data)

    # Benchmark
    time_no_njit = benchmark_function(
        weighted_third_moment, test_data, iterations=100
    )
    time_njit = benchmark_function(
        weighted_third_moment_njit, test_data, iterations=100
    )
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(f"\nweighted_third_moment ({label}):")
    print(f"  Non-njit: {time_no_njit:.4f} ms")
    print(f"  Njit:     {time_njit:.4f} ms")
    print(f"  Speedup:  {speedup:.2f}x")


# ============================================================================
# Tests for fourth_moment
# ============================================================================


def test_fourth_moment_compilation_time():
    """Test compilation time for fourth_moment."""
    test_data = generate_moment_test_data(1000, n=3)

    start = time.perf_counter()
    fourth_moment_njit(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0, "Compilation time should be measurable"
    print(
        f"\nfourth_moment_njit compilation time: {compilation_time * 1000:.2f} ms"
    )


def test_fourth_moment_speedup(test_data_sizes):
    """Test speed-up factor for fourth_moment across different sizes."""
    size = test_data_sizes["size"]
    label = test_data_sizes["name"]

    test_data = generate_moment_test_data(size, n=3)

    # Warm up njit version
    fourth_moment_njit(**test_data)

    # Benchmark
    time_no_njit = benchmark_function(fourth_moment, test_data, iterations=100)
    time_njit = benchmark_function(
        fourth_moment_njit, test_data, iterations=100
    )
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(f"\nfourth_moment ({label}):")
    print(f"  Non-njit: {time_no_njit:.4f} ms")
    print(f"  Njit:     {time_njit:.4f} ms")
    print(f"  Speedup:  {speedup:.2f}x")


# ============================================================================
# Tests for weighted_fourth_moment
# ============================================================================


def test_weighted_fourth_moment_compilation_time():
    """Test compilation time for weighted_fourth_moment."""
    test_data = generate_weighted_test_data(1000)
    test_data["third_moment"] = np.random.randn(1000)

    start = time.perf_counter()
    weighted_fourth_moment_njit(**test_data)
    compilation_time = time.perf_counter() - start

    assert compilation_time > 0, "Compilation time should be measurable"
    print(
        f"\nweighted_fourth_moment_njit compilation time: {compilation_time * 1000:.2f} ms"
    )


def test_weighted_fourth_moment_speedup(test_data_sizes):
    """Test speed-up factor for weighted_fourth_moment across different sizes."""
    size = test_data_sizes["size"]
    label = test_data_sizes["name"]

    if size is None:
        pytest.skip("Weighted functions need array inputs")

    test_data = generate_weighted_test_data(size)
    test_data["third_moment"] = np.random.randn(size)

    # Warm up njit version
    weighted_fourth_moment_njit(**test_data)

    # Benchmark
    time_no_njit = benchmark_function(
        weighted_fourth_moment, test_data, iterations=100
    )
    time_njit = benchmark_function(
        weighted_fourth_moment_njit, test_data, iterations=100
    )
    speedup = calculate_speedup(time_no_njit, time_njit)

    print(f"\nweighted_fourth_moment ({label}):")
    print(f"  Non-njit: {time_no_njit:.4f} ms")
    print(f"  Njit:     {time_njit:.4f} ms")
    print(f"  Speedup:  {speedup:.2f}x")


# ============================================================================
# Summary benchmark test
# ============================================================================


def test_all_functions_summary():
    """Run a comprehensive benchmark of all functions and print a summary."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE NJIT PERFORMANCE BENCHMARK")
    print("=" * 80)

    results = {}

    # Test all functions
    def gen_weighted_third(s):
        data = generate_weighted_test_data(s)
        data["second_moment"] = (
            np.abs(np.random.randn(s)) if s else np.abs(np.random.randn())
        )
        return data

    def gen_weighted_fourth(s):
        data = generate_weighted_test_data(s)
        data["third_moment"] = np.random.randn(s) if s else np.random.randn()
        return data

    functions = [
        (
            "second_moment",
            second_moment,
            second_moment_njit,
            generate_test_data,
        ),
        (
            "weighted_second_moment",
            weighted_second_moment,
            weighted_second_moment_njit,
            generate_weighted_test_data,
        ),
        (
            "third_moment",
            third_moment,
            third_moment_njit,
            lambda s: generate_moment_test_data(s, 2),
        ),
        (
            "weighted_third_moment",
            weighted_third_moment,
            weighted_third_moment_njit,
            gen_weighted_third,
        ),
        (
            "fourth_moment",
            fourth_moment,
            fourth_moment_njit,
            lambda s: generate_moment_test_data(s, 3),
        ),
        (
            "weighted_fourth_moment",
            weighted_fourth_moment,
            weighted_fourth_moment_njit,
            gen_weighted_fourth,
        ),
    ]

    for func_name, func_no_njit, func_njit, data_gen in functions:
        # Use medium array size for summary
        test_data = data_gen(1000)

        # Warm up
        func_njit(**test_data)

        # Benchmark
        time_no_njit = benchmark_function(
            func_no_njit, test_data, iterations=100
        )
        time_njit = benchmark_function(func_njit, test_data, iterations=100)
        speedup = calculate_speedup(time_no_njit, time_njit)

        results[func_name] = {
            "no_njit_ms": time_no_njit,
            "njit_ms": time_njit,
            "speedup": speedup,
        }

    # Print summary table
    print(
        "\n{:<35} {:>15} {:>15} {:>15}".format(
            "Function", "Non-NJIT (ms)", "NJIT (ms)", "Speedup"
        )
    )
    print("-" * 80)

    total_speedup = 0
    for func_name, metrics in results.items():
        print(
            "{:<35} {:>15.4f} {:>15.4f} {:>15.2f}x".format(
                func_name,
                metrics["no_njit_ms"],
                metrics["njit_ms"],
                metrics["speedup"],
            )
        )
        total_speedup += metrics["speedup"]

    avg_speedup = total_speedup / len(results)
    print("-" * 80)
    print(
        "{:<35} {:>31} {:>15.2f}x".format(
            "Average",
            "",
            avg_speedup,
        )
    )
    print("=" * 80 + "\n")
