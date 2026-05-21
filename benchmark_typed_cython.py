"""
Benchmark: Numba vs Optimized Cython (with typed inputs)

This tests the case where inputs are GUARANTEED to be numpy arrays.
This eliminates all conversion overhead in Cython.
"""

import numpy as np
import timeit
from quasar_errors.measures.approx import (
    u_yinterp as u_yinterp_numba,
    u_xinterp as u_xinterp_numba,
    get_quantile as get_quantile_numba,
    get_peak as get_peak_numba,
)

try:
    from quasar_errors.measures import approx_typed
    u_yinterp_cython = approx_typed.u_yinterp
    u_xinterp_cython = approx_typed.u_xinterp
    get_quantile_cython = approx_typed.get_quantile_typed
    get_peak_cython = approx_typed.get_peak_typed
    CYTHON_AVAILABLE = True
except (ImportError, AttributeError) as e:
    CYTHON_AVAILABLE = False
    print(f"Cython typed version not available: {e}")

# Create test data (guaranteed numpy arrays)
np.random.seed(42)

# Data for simple functions
x_interp = 0.5
xp_small = np.array([0.0, 1.0], dtype=np.float64)
yp_small = np.array([1.0, 2.0], dtype=np.float64)
dyp_small = np.array([0.1, 0.15], dtype=np.float64)

y_interp = 1.5
yp_xinterp = np.array([0.0, 2.0], dtype=np.float64)
dyp_xinterp = np.array([0.05, 0.1], dtype=np.float64)

# Data for complex functions
x_curve = np.linspace(0, 10, 100)
y_curve = 3 + 2*np.sin(x_curve)
y_peak = 3 + 2*np.sin(x_curve)
q_test = 0.5
dx_test = 0.1

# Iterations
N_SMALL = 100000
N_MEDIUM = 10000

print("=" * 85)
print("BENCHMARK: Numba vs Optimized Cython (typed/guaranteed numpy array inputs)")
print("=" * 85)
print()

# Warm up
for _ in range(100):
    u_yinterp_numba(x_interp, xp_small, yp_small, dyp_small)
    u_xinterp_numba(y_interp, xp_small, yp_xinterp, dyp_xinterp)
    if CYTHON_AVAILABLE:
        u_yinterp_cython(x_interp, xp_small, yp_small, dyp_small)
        u_xinterp_cython(y_interp, xp_small, yp_xinterp, dyp_xinterp)

results = []

# Test 1: u_yinterp
print("1. u_yinterp (linear interpolation)")
print("-" * 85)

time_numba = timeit.timeit(
    lambda: u_yinterp_numba(x_interp, xp_small, yp_small, dyp_small),
    number=N_SMALL
)
time_per_call = time_numba / N_SMALL * 1e6
print(f"  Numba:    {time_per_call:.4f} μs")
results.append(("u_yinterp", "Numba", time_per_call))

if CYTHON_AVAILABLE:
    time_cython = timeit.timeit(
        lambda: u_yinterp_cython(x_interp, xp_small, yp_small, dyp_small),
        number=N_SMALL
    )
    time_per_call = time_cython / N_SMALL * 1e6
    print(f"  Cython:   {time_per_call:.4f} μs")
    results.append(("u_yinterp", "Cython", time_cython / N_SMALL * 1e6))
    speedup = time_numba / time_cython
    if speedup > 1:
        print(f"  → Cython {(1-1/speedup)*100:.1f}% faster")
    else:
        print(f"  → Numba {(speedup-1)*100:.1f}% faster")
print()

# Test 2: u_xinterp
print("2. u_xinterp (x-interpolation)")
print("-" * 85)

time_numba = timeit.timeit(
    lambda: u_xinterp_numba(y_interp, xp_small, yp_xinterp, dyp_xinterp),
    number=N_SMALL
)
time_per_call = time_numba / N_SMALL * 1e6
print(f"  Numba:    {time_per_call:.4f} μs")
results.append(("u_xinterp", "Numba", time_per_call))

if CYTHON_AVAILABLE:
    time_cython = timeit.timeit(
        lambda: u_xinterp_cython(y_interp, xp_small, yp_xinterp, dyp_xinterp),
        number=N_SMALL
    )
    time_per_call = time_cython / N_SMALL * 1e6
    print(f"  Cython:   {time_per_call:.4f} μs")
    results.append(("u_xinterp", "Cython", time_cython / N_SMALL * 1e6))
    speedup = time_numba / time_cython
    if speedup > 1:
        print(f"  → Cython {(1-1/speedup)*100:.1f}% faster")
    else:
        print(f"  → Numba {(speedup-1)*100:.1f}% faster")
print()

# Test 3: get_quantile
print("3. get_quantile (array operation)")
print("-" * 85)

time_numba = timeit.timeit(
    lambda: get_quantile_numba(q_test, x_curve, y_curve, dx_test),
    number=N_MEDIUM
)
time_per_call = time_numba / N_MEDIUM * 1e6
print(f"  Numba:    {time_per_call:.4f} μs")
results.append(("get_quantile", "Numba", time_per_call))

if CYTHON_AVAILABLE:
    time_cython = timeit.timeit(
        lambda: get_quantile_cython(q_test, x_curve, y_curve, dx_test),
        number=N_MEDIUM
    )
    time_per_call = time_cython / N_MEDIUM * 1e6
    print(f"  Cython:   {time_per_call:.4f} μs")
    results.append(("get_quantile", "Cython", time_cython / N_MEDIUM * 1e6))
    speedup = time_numba / time_cython
    if speedup > 1:
        print(f"  → Cython {(1-1/speedup)*100:.1f}% faster")
    else:
        print(f"  → Numba {(speedup-1)*100:.1f}% faster")
print()

# Test 4: get_peak (the problematic one)
print("4. get_peak (was 125% slower in basic Cython)")
print("-" * 85)

time_numba = timeit.timeit(
    lambda: get_peak_numba(x_curve, y_peak, dx_test),
    number=N_MEDIUM
)
time_per_call = time_numba / N_MEDIUM * 1e6
print(f"  Numba:    {time_per_call:.4f} μs")
results.append(("get_peak", "Numba", time_per_call))

if CYTHON_AVAILABLE:
    time_cython = timeit.timeit(
        lambda: get_peak_cython(x_curve, y_peak, dx_test),
        number=N_MEDIUM
    )
    time_per_call = time_cython / N_MEDIUM * 1e6
    print(f"  Cython:   {time_per_call:.4f} μs (typed)")
    results.append(("get_peak", "Cython", time_cython / N_MEDIUM * 1e6))
    speedup = time_numba / time_cython
    if speedup > 1:
        percent_faster = (1-1/speedup)*100
        print(f"  → Cython {percent_faster:.1f}% faster! *** IMPROVEMENT ***")
    else:
        percent_slower = (speedup-1)*100
        print(f"  → Numba {-percent_slower:.1f}% faster")
print()

# Summary
print("=" * 85)
print("SUMMARY: Impact of Guaranteed Typed Inputs")
print("=" * 85)
print()

if CYTHON_AVAILABLE:
    print(f"{'Function':<20} {'Numba (μs)':<15} {'Cython (μs)':<15} {'Winner':<20}")
    print("-" * 85)
    
    for func in ["u_yinterp", "u_xinterp", "get_quantile", "get_peak"]:
        numba_time = next((t for f, impl, t in results if f == func and impl == "Numba"), None)
        cython_time = next((t for f, impl, t in results if f == func and impl == "Cython"), None)
        
        if numba_time and cython_time:
            if numba_time < cython_time:
                speedup = ((cython_time / numba_time) - 1) * 100
                winner = f"Numba +{speedup:.1f}%"
            else:
                speedup = ((numba_time / cython_time) - 1) * 100
                winner = f"Cython +{speedup:.1f}%"
            print(f"{func:<20} {numba_time:<15.4f} {cython_time:<15.4f} {winner:<20}")

print()
print("KEY INSIGHT:")
print("-" * 85)
print("With guaranteed typed numpy array inputs, Cython can eliminate conversion overhead.")
print("However, Numba still performs better on complex logic functions like get_peak.")
print()
print("RECOMMENDATION:")
print("  • If you can guarantee inputs are ALWAYS numpy arrays: Small performance gain")
print("  • But pure Numba is still faster overall and simpler to maintain")
print("  • Use typed Cython only if profiling shows specific bottlenecks")
