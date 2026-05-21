__all__ = ["nth_moment"]

from typing import Optional
from quasar_typing.numpy import FloatVector, FloatMatrix

from .measure_result import MeasureResult

def _total(
    ys: FloatMatrix, 
    dxs: FloatVector, 
    dys: FloatMatrix,
) -> tuple[FloatVector, FloatVector]:
    r"""    
    Calculate total integrated flux and its uncertainty.
    
    Notes
    -----
    **Value**: :math:`T = \sum_i y_i \Delta x_i`
    
    **Uncertainty**:
    :math:`\sigma_T = \sqrt{\sum_i (\sigma_{y_i} \Delta x_i)^2}`
    
    The total is a weighted sum where each flux measurement is weighted
    by its wavelength bin width. The uncertainty propagates linearly
    since each term appears only once in the sum.
    """
    return (
        (ys * dxs).sum(axis=-1),
        (dys**2 * dxs**2).sum(axis=-1)**0.5,
    )

def _nth_moment(
    n: int, 
    xs: FloatMatrix, 
    ys: FloatMatrix, 
    dxs: FloatMatrix, 
    dys: FloatMatrix, 
    tot: Optional[tuple[FloatVector, FloatVector]] = None,
) -> tuple[FloatVector, FloatVector]:
    r"""
    Returns the flux-weighted nth moment and its uncertainty.
    
    Notes
    -----
    **Value**:
    :math:`\mu_n = \frac{\sum_i y_i \lambda_i^n \Delta x_i}{T}`
    
    **Uncertainty**:
    :math:`\sigma_{\mu_n} = \frac{\sqrt{\sum_i (\sigma_{y_i}
    \Delta x_i (\lambda_i^n - \mu_n))^2}}{T}`
    
    The nth moment is the flux-weighted average of wavelength powers.
    The uncertainty propagates through numerator and denominator via
    first-order Taylor expansion.
    """
    if tot is None:
        tot = _total(ys, dxs, dys)

    if n == 0:
        return tot

    val = (ys * xs**n * dxs).sum(axis=-1) / tot[0]
    ds = xs**n - val[:,None]
    unc = (dys**2 * dxs**2 * ds**2).sum(axis=-1)**0.5 / tot[0]

    return (val, unc)

def _nth_central_moment(
    n: int, 
    xs: FloatMatrix, 
    ys: FloatMatrix, 
    dxs: FloatMatrix, 
    dys: FloatMatrix, 
    tot: Optional[tuple[FloatVector, FloatVector]] = None, 
    m1: Optional[tuple[FloatVector, FloatVector]] = None,
) -> tuple[FloatVector, FloatVector]:
    r"""
    Returns the nth central moment and its uncertainty.
    
    Notes
    -----
    **Value**:
    :math:`\mu_n = \frac{\sum_i y_i (\lambda_i - \mu_1)^n \Delta x_i}{T}`
    
    **Uncertainty**:
    :math:`\sigma_{\mu_n} = \frac{\sqrt{\sum_i (\sigma_{y_i} \Delta x_i
    [(\lambda_i - \mu_1)^n - n(\lambda_i - \mu_1)\mu_{n-1}
    - \mu_n])^2}}{T}`
    
    Central moments measure variance around mean wavelength. Uncertainty
    accounts for dependence on :math:`\mu_1` through the chain rule.
    """
    if n == 0:
        return _total(ys, dxs, dys)
    if n == 1:
        return _nth_moment(1, xs, ys, dxs, dys, tot=tot)

    if tot is None:
        tot = _total(ys, dxs, dys)
    if m1 is None:
        m1 = _nth_moment(1, xs, ys, dxs, dys, tot=tot)

    diffs = (xs - m1[0][:,None]) 

    val = (ys * diffs**n * dxs).sum(axis=-1) / tot[0]
    mu_nm1 = (ys * diffs**(n-1) * dxs).sum(axis=-1) / tot[0]
    ds = diffs**n - n * diffs * mu_nm1[:,None] - val[:,None]
    unc = (dys**2 * dxs**2 * ds**2).sum(axis=-1)**0.5 / tot[0]

    return (val, unc)


def _dispersion(
    xs: FloatMatrix, 
    ys: FloatMatrix, 
    dxs: FloatMatrix, 
    dys: FloatMatrix, 
    tot: Optional[tuple[FloatVector, FloatVector]] = None, 
    m1: Optional[tuple[FloatVector, FloatVector]] = None,
    cm2: Optional[tuple[FloatVector, FloatVector]] = None,
) -> tuple[FloatVector, FloatVector]:
    r"""
    Returns relative dispersion sqrt(mu2) / mu1 and its uncertainty.
    
    Notes
    -----
    **Value**: :math:`d = \frac{\sqrt{\mu_2}}{\mu_1}`
    
    **Uncertainty**:
    :math:`\sigma_d = \frac{\sqrt{\sum_i (\sigma_{y_i} \Delta x_i)^2
    [\frac{\lambda_i^2 - \mu_2}{2\sqrt{\mu_2}} -
    \frac{\sqrt{\mu_2}(\lambda_i - \mu_1)}{\mu_1}]^2}}{T}`
    
    Dispersion measures fractional wavelength spread. The uncertainty
    propagates through numerator and denominator via quotient rule.
    """
    if tot is None:
        tot = _total(ys, dxs, dys)
    if m1 is None:
        m1 = _nth_moment(1, xs, ys, dxs, dys, tot=tot)
    if cm2 is None:
        cm2 = _nth_central_moment(2, xs, ys, dxs, dys, tot=tot, m1=m1)

    num = cm2[0]**0.5
    den = m1[0]
    val = num / den
    
    diffs = xs - m1[0][:,None]
    dmu2_deriv = diffs**2 - cm2[0][:,None]
    d_dmu2 = dmu2_deriv / (2.0 * num[:,None])
    d_mu2 = num[:,None] * diffs / den[:,None]
    ds = (d_dmu2 - d_mu2) / den[:,None]
    
    unc = (dys**2 * dxs**2 * ds**2).sum(axis=-1)**0.5 / tot[0]

    return (val, unc)

def _skewness(
    xs: FloatMatrix, 
    ys: FloatMatrix, 
    dxs: FloatMatrix, 
    dys: FloatMatrix, 
    tot: Optional[tuple[FloatVector, FloatVector]] = None, 
    m1: Optional[tuple[FloatVector, FloatVector]] = None, 
    cm2: Optional[tuple[FloatVector, FloatVector]] = None, 
    cm3: Optional[tuple[FloatVector, FloatVector]] = None,
) -> tuple[FloatVector, FloatVector]:
    r"""
    Returns skewness mu3 / mu2^(3/2) and its uncertainty.
    
    Notes
    -----
    **Value**: :math:`\gamma = \frac{\mu_3}{\mu_2^{3/2}}`
    
    **Moment Derivatives** (w.r.t. :math:`y_i`):
    
    .. math::
    
        \frac{\partial \mu_2}{\partial y_i} = \frac{\Delta x_i}{T}
        [(\lambda_i - \mu_1)^2 - \mu_2]
        
        \frac{\partial \mu_3}{\partial y_i} = \frac{\Delta x_i}{T}
        [(\lambda_i - \mu_1)^3 - \mu_3 - 3\mu_2(\lambda_i - \mu_1)]
    
    **Skewness Partials** (w.r.t. moments):
    
    .. math::
    
        \frac{\partial \gamma}{\partial \mu_2} = -\frac{3\mu_3}{2\mu_2^{5/2}}
        
        \frac{\partial \gamma}{\partial \mu_3} = \frac{1}{\mu_2^{3/2}}
    
    **Uncertainty**:
    :math:`\sigma_\gamma = \sqrt{\sum_i \sigma_{y_i}^2
    (\partial \gamma / \partial y_i)^2}`
    
    The :math:`-3\mu_2(\lambda_i - \mu_1)` term in
    :math:`\partial \mu_3 / \partial y_i` is critical for non-Gaussian
    spectral shapes and ensures correct uncertainty propagation across
    symmetric and skewed distributions.
    """
    if tot is None:
        tot = _total(ys, dxs, dys)
    if m1 is None:
        m1 = _nth_moment(1, xs, ys, dxs, dys, tot=tot)
    if cm2 is None:
        cm2 = _nth_central_moment(2, xs, ys, dxs, dys, tot=tot, m1=m1)
    if cm3 is None:
        cm3 = _nth_central_moment(3, xs, ys, dxs, dys, tot=tot, m1=m1)

    num = cm3[0]
    den = cm2[0]**1.5
    val = num / den

    diffs = xs - m1[0][:,None]
    
    dm2 = (diffs**2 - cm2[0][:,None]) * dxs / tot[0][:,None]
    dm3 = (diffs**3 - cm3[0][:,None] - 3*cm2[0][:,None] * diffs) * dxs \
        / tot[0][:,None]
    
    d_dm2 = -1.5 * cm3[0] / (cm2[0]**2.5)
    d_dm3 = 1.0 / (cm2[0]**1.5)
    
    d_dy = d_dm3[:,None] * dm3 + d_dm2[:,None] * dm2
    
    unc = (dys**2 * d_dy**2).sum(axis=-1)**0.5

    return (val, unc)

def _kurtosis(
    xs: FloatMatrix, 
    ys: FloatMatrix, 
    dxs: FloatMatrix, 
    dys: FloatMatrix, 
    tot: Optional[tuple[FloatVector, FloatVector]] = None, 
    m1: Optional[tuple[FloatVector, FloatVector]] = None, 
    cm2: Optional[tuple[FloatVector, FloatVector]] = None, 
    cm3: Optional[tuple[FloatVector, FloatVector]] = None, 
    cm4: Optional[tuple[FloatVector, FloatVector]] = None,
) -> tuple[FloatVector, FloatVector]:
    """
    Returns the kurtosis mu4 / mu2^2 and its uncertainty.
    """
    if tot is None:
        tot = _total(ys, dxs, dys)
    if m1 is None:
        m1 = _nth_moment(1, xs, ys, dxs, dys, tot=tot)
    if cm2 is None:
        cm2 = _nth_central_moment(2, xs, ys, dxs, dys, tot=tot, m1=m1)
    if cm3 is None:
        cm3 = _nth_central_moment(3, xs, ys, dxs, dys, tot=tot, m1=m1)
    if cm4 is None:
        cm4 = _nth_central_moment(4, xs, ys, dxs, dys, tot=tot, m1=m1)

    num = cm4[0]
    den = cm2[0]**2
    val = num / den

    diffs = xs - m1[0][:,None]
    dmu4 = diffs**4 - 4.0 * diffs * cm3[0][:,None] - cm4[0][:,None]
    dmu2 = diffs**2 - cm2[0][:,None]
    ds = dmu4 / den[:,None] - 2.0 * val[:,None] * dmu2 / cm2[0][:,None]
    unc = (dys**2 * dxs**2 * ds**2).sum(axis=-1)**0.5 / tot[0]

    return (val, unc)

###

def total(
    ys:  FloatMatrix,
    dxs: FloatVector | FloatMatrix,
    dys: FloatVector | FloatMatrix,
) -> MeasureResult:
    if dys.ndim == 1:
        dys = dys[None,:]
    if dxs.ndim == 1:
        dxs = dxs[None,:]
    return MeasureResult(*_total(ys, dxs, dys))

def nth_moment(
    n: int,
    xs: FloatVector | FloatMatrix,
    ys:  FloatMatrix,
    dxs: FloatVector | FloatMatrix,
    dys: FloatVector | FloatMatrix,
    *,
    tot: MeasureResult | None = None,
) -> MeasureResult:
    if xs.ndim == 1:
        xs = xs[None,:]
    if dys.ndim == 1:
        dys = dys[None,:]
    if dxs.ndim == 1:
        dxs = dxs[None,:]
    return MeasureResult(*_nth_moment(
        n, 
        xs, ys, dxs, dys, 
        tot=tot if (tot is None) else tot.to_tuple(),
    ))

def nth_central_moment(
    n: int,
    xs: FloatVector | FloatMatrix,
    ys:  FloatMatrix,
    dxs: FloatVector | FloatMatrix,
    dys: FloatVector | FloatMatrix,
    *,
    tot: MeasureResult | None = None,
    m1: MeasureResult | None = None,
) -> MeasureResult:
    if xs.ndim == 1:
        xs = xs[None,:]
    if dys.ndim == 1:
        dys = dys[None,:]
    if dxs.ndim == 1:
        dxs = dxs[None,:]
    return MeasureResult(*_nth_central_moment(
        n, 
        xs, ys, dxs, dys, 
        tot=tot if (tot is None) else tot.to_tuple(), 
        m1=m1 if (m1 is None) else m1.to_tuple(),
    ))

def dispersion(
    xs: FloatVector | FloatMatrix,
    ys:  FloatMatrix,
    dxs: FloatVector | FloatMatrix,
    dys: FloatVector | FloatMatrix,
    *,
    tot: MeasureResult | None = None,
    m1: MeasureResult | None = None,
    cm2: MeasureResult | None = None,
) -> MeasureResult:
    if xs.ndim == 1:
        xs = xs[None,:]
    if dys.ndim == 1:
        dys = dys[None,:]
    if dxs.ndim == 1:
        dxs = dxs[None,:]
    return MeasureResult(*_dispersion(
        xs, ys, dxs, dys, 
        tot=tot if (tot is None) else tot.to_tuple(), 
        m1=m1 if (m1 is None) else m1.to_tuple(), 
        cm2=cm2 if (cm2 is None) else cm2.to_tuple(),
    ))

def skewness(
    xs: FloatVector | FloatMatrix,
    ys:  FloatMatrix,
    dxs: FloatVector | FloatMatrix,
    dys: FloatVector | FloatMatrix,
    *,
    tot: MeasureResult | None = None,
    m1: MeasureResult | None = None,
    cm2: MeasureResult | None = None,
    cm3: MeasureResult | None = None,
) -> MeasureResult:
    if xs.ndim == 1:
        xs = xs[None,:]
    if dys.ndim == 1:
        dys = dys[None,:]
    if dxs.ndim == 1:
        dxs = dxs[None,:]
    val, unc = _skewness(
        xs, ys, dxs, dys, 
        tot=tot if (tot is None) else tot.to_tuple(), 
        m1=m1 if (m1 is None) else m1.to_tuple(), 
        cm2=cm2 if (cm2 is None) else cm2.to_tuple(), 
        cm3=cm3 if (cm3 is None) else cm3.to_tuple(),
    )
    print(val.shape, unc.shape)
    return MeasureResult(val, unc=unc)

def kurtosis(
    xs: FloatVector | FloatMatrix,
    ys:  FloatMatrix,
    dxs: FloatVector | FloatMatrix,
    dys: FloatVector | FloatMatrix,
    *,
    tot: MeasureResult | None = None,
    m1: MeasureResult | None = None,
    cm2: MeasureResult | None = None,
    cm3: MeasureResult | None = None,
    cm4: MeasureResult | None = None,
) -> MeasureResult:
    if xs.ndim == 1:
        xs = xs[None,:]
    if dys.ndim == 1:
        dys = dys[None,:]
    if dxs.ndim == 1:
        dxs = dxs[None,:]
    return MeasureResult(*_kurtosis(
        xs, ys, dxs, dys, 
        tot=tot if (tot is None) else tot.to_tuple(), 
        m1=m1 if (m1 is None) else m1.to_tuple(), 
        cm2=cm2 if (cm2 is None) else cm2.to_tuple(), 
        cm3=cm3 if (cm3 is None) else cm3.to_tuple(), 
        cm4=cm4 if (cm4 is None) else cm4.to_tuple(),
    ))