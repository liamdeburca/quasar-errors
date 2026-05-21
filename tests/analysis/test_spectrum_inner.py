"""
Validate spectrum_inner moment calculations against the `uncertainties` package.

Pure-numpy reimplementations of the formulas (identical math, no numba) are
tested against `uncertainties`, which tracks all correlations exactly via
first-order error propagation.
"""
import numpy as np
import pytest
from uncertainties import unumpy


# ---------------------------------------------------------------------------
# Pure-numpy mirror of spectrum_inner.py
# ---------------------------------------------------------------------------
def total(ys, dys, dxs):
    return (
        (ys * dxs).sum(axis=1),
        (dys**2 * dxs**2).sum(axis=1)**0.5,
    )


def nth_moment(n, xs, ys, dys, dxs, *, tot=None):
    if tot is None:
        tot = total(ys, dys, dxs)
    if n == 0:
        return tot
    val = (ys * xs**n * dxs).sum(axis=1) / tot[0]
    unc = (dys**2 * dxs**2 * (xs**n - val[:, None])**2).sum(axis=1)**0.5 / tot[0]
    return (val, unc)


def nth_central_moment(n, xs, ys, dys, dxs, *, tot=None, m1=None):
    if xs.ndim == 1:
        xs = xs[None, :]
    if dys.ndim == 1:
        dys = dys[None, :]
    if dxs.ndim == 1:
        dxs = dxs[None, :]

    if n == 0:
        return total(ys, dys, dxs)
    if n == 1:
        return nth_moment(1, xs, ys, dys, dxs, tot=tot)

    if tot is None:
        tot = total(ys, dys, dxs)
    if m1 is None:
        m1 = nth_moment(1, xs, ys, dys, dxs, tot=tot)

    diffs = xs - m1[0][:, None]

    val = (ys * diffs**n * dxs).sum(axis=1) / tot[0]
    mu_nm1 = (ys * diffs**(n - 1) * dxs).sum(axis=1) / tot[0]
    ds = diffs**n - n * diffs * mu_nm1[:, None] - val[:, None]
    unc = (dys**2 * dxs**2 * ds**2).sum(axis=1)**0.5 / tot[0]
    return (val, unc)


def skewness(xs, ys, dys, dxs):
    xs = xs[None, :] if xs.ndim == 1 else xs
    dys = dys[None, :] if dys.ndim == 1 else dys
    dxs = dxs[None, :] if dxs.ndim == 1 else dxs

    tot = total(ys, dys, dxs)
    m1 = nth_moment(1, xs, ys, dys, dxs, tot=tot)
    m2 = nth_central_moment(2, xs, ys, dys, dxs, tot=tot, m1=m1)
    m3 = nth_central_moment(3, xs, ys, dys, dxs, tot=tot, m1=m1)

    num = m3[0]
    den = m2[0]**1.5
    val = num / den

    diffs = xs - m1[0][:, None]
    T = tot[0]
    
    # Correct moment derivatives: ∂μ₂/∂yᵢ = dxs * ((λᵢ - μ₁)² - μ₂) / T
    # (derivatives w.r.t. y account for flux integration weight dxs)
    dmu2 = (diffs**2 - m2[0][:, None]) * dxs / T
    dmu3 = (diffs**3 - m3[0][:, None] - 3*m2[0][:, None]*diffs) * dxs / T
    
    # Skewness partial derivatives
    dgamma_dmu2 = -1.5 * m3[0] / (m2[0]**2.5)
    dgamma_dmu3 = 1.0 / (m2[0]**1.5)
    
    # Chain rule: ∂γ/∂yᵢ
    dgamma_dyi = dgamma_dmu3[:, None] * dmu3 + dgamma_dmu2[:, None] * dmu2
    
    # Uncertainty propagation
    unc = (dys**2 * dgamma_dyi**2).sum(axis=1)**0.5

    return (val, unc)


def kurtosis(xs, ys, dys, dxs):
    xs = xs[None, :] if xs.ndim == 1 else xs
    dys = dys[None, :] if dys.ndim == 1 else dys
    dxs = dxs[None, :] if dxs.ndim == 1 else dxs

    tot = total(ys, dys, dxs)
    m1 = nth_moment(1, xs, ys, dys, dxs, tot=tot)
    m2 = nth_central_moment(2, xs, ys, dys, dxs, tot=tot, m1=m1)
    m4 = nth_central_moment(4, xs, ys, dys, dxs, tot=tot, m1=m1)

    mu2 = m2[0]
    mu4 = m4[0]

    diffs = xs - m1[0][:, None]
    mu3 = (ys * diffs**3 * dxs).sum(axis=1) / tot[0]

    num = mu4
    den = mu2**2
    val = num / den

    dmu4 = diffs**4 - 4.0 * diffs * mu3[:, None] - mu4[:, None]
    dmu2 = diffs**2 - mu2[:, None]
    ds = dmu4 / den[:, None] - 2.0 * val[:, None] * dmu2 / mu2[:, None]
    unc = (dys**2 * dxs**2 * ds**2).sum(axis=1)**0.5 / tot[0]

    return (val, unc)


def dispersion(xs, ys, dys, dxs):
    xs = xs[None, :] if xs.ndim == 1 else xs
    dys = dys[None, :] if dys.ndim == 1 else dys
    dxs = dxs[None, :] if dxs.ndim == 1 else dxs

    tot = total(ys, dys, dxs)
    m1 = nth_moment(1, xs, ys, dys, dxs, tot=tot)
    m2 = nth_central_moment(2, xs, ys, dys, dxs, tot=tot, m1=m1)

    num = m2[0]**0.5
    den = m1[0]
    val = num / den

    diffs = xs - m1[0][:, None]
    dmu2_deriv = diffs**2 - m2[0][:, None]
    ds = (dmu2_deriv / (2.0 * num[:, None]) - num[:, None] * diffs / den[:, None]) / den[:, None]
    unc = (dys**2 * dxs**2 * ds**2).sum(axis=1)**0.5 / tot[0]

    return (val, unc)


# ---------------------------------------------------------------------------
# Reference implementation using the uncertainties package
# ---------------------------------------------------------------------------
def uncertainties_total(ys_u, dxs):
    return np.sum(ys_u * dxs)


def uncertainties_nth_moment(n, xs, ys_u, dxs):
    T = np.sum(ys_u * dxs)
    if n == 0:
        return T
    return np.sum(ys_u * xs**n * dxs) / T


def uncertainties_nth_central_moment(n, xs, ys_u, dxs):
    T = np.sum(ys_u * dxs)
    if n == 0:
        return T
    m1 = np.sum(ys_u * xs * dxs) / T
    if n == 1:
        return m1
    return np.sum(ys_u * (xs - m1)**n * dxs) / T

def uncertainties_dispersion(xs, ys_u, dxs):
    T = np.sum(ys_u * dxs)
    m1 = np.sum(ys_u * xs * dxs) / T
    mu2 = np.sum(ys_u * (xs - m1)**2 * dxs) / T
    return mu2**0.5 / m1


def uncertainties_skewness(xs, ys_u, dxs):
    T = np.sum(ys_u * dxs)
    m1 = np.sum(ys_u * xs * dxs) / T
    mu2 = np.sum(ys_u * (xs - m1)**2 * dxs) / T
    mu3 = np.sum(ys_u * (xs - m1)**3 * dxs) / T
    return mu3 / mu2**1.5


def uncertainties_kurtosis(xs, ys_u, dxs):
    T = np.sum(ys_u * dxs)
    m1 = np.sum(ys_u * xs * dxs) / T
    mu2 = np.sum(ys_u * (xs - m1)**2 * dxs) / T
    mu4 = np.sum(ys_u * (xs - m1)**4 * dxs) / T
    return mu4 / mu2**2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=[
    "narrow_gaussian",
    "wide_gaussian",
    "bimodal",
    "skewed_left",
    "skewed_right",
    "uniform_like",
    "exponential_like",
    "small_sample_high_unc",
    "large_sample_low_unc",
    "double_peak_asymmetric",
])
def spectrum(request):
    rng = np.random.default_rng(42)
    spec_type = request.param
    
    if spec_type == "narrow_gaussian":
        N = 50
        xs = np.linspace(4800, 5200, N)
        dxs = np.full(N, xs[1] - xs[0])
        ys = 10.0 * np.exp(-0.5 * ((xs - 5000) / 50) ** 2) + 1.0
        dys = 0.3 + 0.1 * rng.random(N)
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    elif spec_type == "wide_gaussian":
        N = 50
        xs = np.linspace(4700, 5300, N)
        dxs = np.full(N, xs[1] - xs[0])
        ys = 8.0 * np.exp(-0.5 * ((xs - 5000) / 150) ** 2) + 0.5
        dys = 0.2 + 0.08 * rng.random(N)
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    elif spec_type == "bimodal":
        N = 50
        xs = np.linspace(4800, 5200, N)
        dxs = np.full(N, xs[1] - xs[0])
        ys = (5.0 * np.exp(-0.5 * ((xs - 4900) / 40) ** 2) + 
              5.0 * np.exp(-0.5 * ((xs - 5100) / 40) ** 2) + 0.5)
        dys = 0.25 + 0.1 * rng.random(N)
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    elif spec_type == "skewed_left":
        N = 50
        xs = np.linspace(4800, 5200, N)
        dxs = np.full(N, xs[1] - xs[0])
        # Skewed toward lower wavelengths
        ys = 12.0 * np.exp(-0.5 * ((xs - 4850) / 80) ** 2) + 0.5
        dys = 0.3 + 0.12 * rng.random(N)
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    elif spec_type == "skewed_right":
        N = 50
        xs = np.linspace(4800, 5200, N)
        dxs = np.full(N, xs[1] - xs[0])
        # Skewed toward higher wavelengths
        ys = 12.0 * np.exp(-0.5 * ((xs - 5150) / 80) ** 2) + 0.5
        dys = 0.3 + 0.12 * rng.random(N)
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    elif spec_type == "uniform_like":
        N = 50
        xs = np.linspace(4800, 5200, N)
        dxs = np.full(N, xs[1] - xs[0])
        ys = 5.0 + 2.0 * np.sin(np.pi * (xs - 4800) / 400) + 0.5 * rng.random(N)
        dys = 0.4 + 0.15 * rng.random(N)
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    elif spec_type == "exponential_like":
        N = 50
        xs = np.linspace(4800, 5200, N)
        dxs = np.full(N, xs[1] - xs[0])
        ys = 15.0 * np.exp(-0.008 * (xs - 4800)) + 0.5
        dys = 0.35 + 0.15 * rng.random(N)
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    elif spec_type == "small_sample_high_unc":
        N = 20
        xs = np.linspace(4800, 5200, N)
        dxs = np.full(N, xs[1] - xs[0])
        ys = 8.0 * np.exp(-0.5 * ((xs - 5000) / 60) ** 2) + 1.0
        dys = 0.8 + 0.3 * rng.random(N)  # High relative uncertainty
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    elif spec_type == "large_sample_low_unc":
        N = 100
        xs = np.linspace(4800, 5200, N)
        dxs = np.full(N, xs[1] - xs[0])
        ys = 10.0 * np.exp(-0.5 * ((xs - 5000) / 50) ** 2) + 1.0
        dys = 0.1 + 0.03 * rng.random(N)  # Low relative uncertainty
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    elif spec_type == "double_peak_asymmetric":
        N = 50
        xs = np.linspace(4800, 5200, N)
        dxs = np.full(N, xs[1] - xs[0])
        # Strong peak on left, weak peak on right
        ys = (8.0 * np.exp(-0.5 * ((xs - 4900) / 35) ** 2) + 
              2.0 * np.exp(-0.5 * ((xs - 5150) / 60) ** 2) + 0.5)
        dys = 0.28 + 0.11 * rng.random(N)
        ys = ys + rng.normal(0, dys, N)  # Add Gaussian noise
    
    return xs, ys, dys, dxs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestTotal:
    def test_value(self, spectrum):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        code_val, _ = total(ys[None, :], dys[None, :], dxs[None, :])
        ref = uncertainties_total(ys_u, dxs)

        assert code_val[0] == pytest.approx(ref.nominal_value, rel=1e-12)

    def test_uncertainty(self, spectrum):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        _, code_unc = total(ys[None, :], dys[None, :], dxs[None, :])
        ref = uncertainties_total(ys_u, dxs)

        assert code_unc[0] == pytest.approx(ref.std_dev, rel=1e-12)


class TestNthMoment:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_value(self, spectrum, n):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        code_val, _ = nth_moment(
            n, xs[None, :], ys[None, :], dys[None, :], dxs[None, :],
        )
        ref = uncertainties_nth_moment(n, xs, ys_u, dxs)

        assert code_val[0] == pytest.approx(ref.nominal_value, rel=1e-10)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_uncertainty(self, spectrum, n):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        _, code_unc = nth_moment(
            n, xs[None, :], ys[None, :], dys[None, :], dxs[None, :],
        )
        ref = uncertainties_nth_moment(n, xs, ys_u, dxs)

        assert code_unc[0] == pytest.approx(ref.std_dev, rel=1e-10)


class TestNthCentralMoment:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_value(self, spectrum, n):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        code_val, _ = nth_central_moment(
            n, xs[None, :], ys[None, :], dys[None, :], dxs[None, :],
        )
        ref = uncertainties_nth_central_moment(n, xs, ys_u, dxs)

        assert code_val[0] == pytest.approx(ref.nominal_value, rel=1e-10, abs=1e-6)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_uncertainty(self, spectrum, n):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        _, code_unc = nth_central_moment(
            n, xs[None, :], ys[None, :], dys[None, :], dxs[None, :],
        )
        ref = uncertainties_nth_central_moment(n, xs, ys_u, dxs)

        assert code_unc[0] == pytest.approx(ref.std_dev, rel=1e-10)


class TestDispersion:
    def test_value(self, spectrum):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        code_val, _ = dispersion(xs, ys[None, :], dys, dxs)
        ref = uncertainties_dispersion(xs, ys_u, dxs)

        assert code_val[0] == pytest.approx(ref.nominal_value, rel=1e-10)

    def test_uncertainty(self, spectrum):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        _, code_unc = dispersion(xs, ys[None, :], dys, dxs)
        ref = uncertainties_dispersion(xs, ys_u, dxs)

        assert code_unc[0] == pytest.approx(ref.std_dev, rel=1e-10)


class TestSkewness:
    def test_value(self, spectrum):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        code_val, _ = skewness(xs, ys[None, :], dys, dxs)
        ref = uncertainties_skewness(xs, ys_u, dxs)

        assert code_val[0] == pytest.approx(ref.nominal_value, rel=1e-10, abs=1e-6)

    def test_uncertainty(self, spectrum):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        _, code_unc = skewness(xs, ys[None, :], dys, dxs)
        ref = uncertainties_skewness(xs, ys_u, dxs)

        assert code_unc[0] == pytest.approx(ref.std_dev, rel=1e-2)


class TestKurtosis:
    def test_value(self, spectrum):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        code_val, _ = kurtosis(xs, ys[None, :], dys, dxs)
        ref = uncertainties_kurtosis(xs, ys_u, dxs)

        assert code_val[0] == pytest.approx(ref.nominal_value, rel=1e-10)

    def test_uncertainty(self, spectrum):
        xs, ys, dys, dxs = spectrum
        ys_u = unumpy.uarray(ys, dys)

        _, code_unc = kurtosis(xs, ys[None, :], dys, dxs)
        ref = uncertainties_kurtosis(xs, ys_u, dxs)

        assert code_unc[0] == pytest.approx(ref.std_dev, rel=1e-10)