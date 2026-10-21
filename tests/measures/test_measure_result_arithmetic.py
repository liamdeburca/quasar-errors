"""
Comprehensive test suite for MeasureResult and FrozenMeasureResult arithmetic operations.

Uses the `uncertainties` module as ground truth for error propagation validation.

Tests cover:
- Arithmetic with floats, ints, Quantities, and MeasureResult instances
- Unit handling and combinations
- Uncertainty propagation (validated against uncertainties module)
- Both mutable (MeasureResult) and immutable (FrozenMeasureResult) classes
- In-place operations
"""

from astropy.units import Quantity, Unit, dimensionless_unscaled
from numpy import allclose, array, isfinite, isnan, nan
from uncertainties.unumpy import uarray

from quasar_errors.analysis.measure_result import (
    FrozenMeasureResult,
    MeasureResult,
)


def test_add_float():
    """Test adding a float to MeasureResult. Validated against uncertainties module."""
    val = array([1.0, 2.0, 3.0])
    unc = array([0.1, 0.1, 0.1])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    # Compute expected values using uncertainties module
    expected = uarray(val, unc) + 5.0
    
    result = mr + 5.0

    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")

def test_add_int():
    """Test adding an int to MeasureResult. Validated against uncertainties module."""
    val = array([1.0, 2.0, 3.0])
    unc = array([0.1, 0.1, 0.1])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) + 5
    
    result = mr + 5
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")

def test_sub_float():
    """Test subtracting a float from MeasureResult. Validated against uncertainties module."""
    val = array([5.0, 6.0, 7.0])
    unc = array([0.1, 0.1, 0.1])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) - 2.0
    
    result = mr - 2.0
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")

def test_mul_float():
    """Test multiplying MeasureResult by float. Validated against uncertainties module."""
    val = array([2.0, 3.0, 4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) * 2.5
    
    result = mr * 2.5
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")

def test_mul_negative_float():
    """Test multiplying by negative float. Validated against uncertainties module."""
    val = array([2.0, 3.0, 4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) * (-2.5)
    
    result = mr * (-2.5)
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")

def test_rmul_float():
    """Test multiplying float by MeasureResult (reverse multiply). Validated against uncertainties module."""
    val = array([2.0, 3.0, 4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = 2.5 * uarray(val, unc)
    
    result = 2.5 * mr
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")

def test_truediv_float():
    """Test dividing MeasureResult by float. Validated against uncertainties module."""
    val = array([10.0, 20.0, 30.0])
    unc = array([1.0, 2.0, 3.0])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) / 2.0
    
    result = mr / 2.0
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")

def test_rtruediv_float():
    """Test dividing float by MeasureResult (reverse divide). Validated against uncertainties module."""
    val = array([2.0, 4.0, 5.0])
    unc = array([0.1, 0.2, 0.25])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = 10.0 / uarray(val, unc)
    
    result = 10.0 / mr
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("1/m")

def test_pow_float():
    """Test raising MeasureResult to a power. Validated against uncertainties module."""
    val = array([2.0, 3.0, 4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) ** 2.0
    
    result = mr ** 2.0
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m2")

def test_pow_zero():
    """Test raising MeasureResult to power 0."""
    val = array([2.0, 3.0, 4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    result = mr ** 0
    
    assert allclose(result.val, [1.0, 1.0, 1.0])
    assert allclose(result.unc, [0.0, 0.0, 0.0])
    assert result.unit == dimensionless_unscaled

def test_abs():
    """Test absolute value of MeasureResult."""
    val = array([-2.0, 3.0, -4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    result = abs(mr)
    
    assert allclose(result.val, [2.0, 3.0, 4.0])
    assert allclose(result.unc, [0.1, 0.2, 0.3])
    assert result.unit == Unit("m")


def test_mul_quantity():
    """Test multiplying MeasureResult by Quantity. Validated against uncertainties module."""
    val = array([2.0, 3.0, 4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    q = Quantity(5.0, "s")
    # For Quantity multiplication, we use the Quantity value
    expected = uarray(val, unc) * q.value
    
    result = mr * q
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m*s")

def test_mul_quantity_negative():
    """Test multiplying by negative Quantity. Validated against uncertainties module."""
    val = array([2.0, 3.0, 4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    q = Quantity(-5.0, "s")
    expected = uarray(val, unc) * q.value
    
    result = mr * q
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m*s")

def test_truediv_quantity():
    """Test dividing MeasureResult by Quantity. Validated against uncertainties module."""
    val = array([10.0, 20.0, 30.0])
    unc = array([1.0, 2.0, 3.0])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    q = Quantity(2.0, "s")
    expected = uarray(val, unc) / q.value
    
    result = mr / q
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m/s")

def test_add_measure_results():
    """Test adding two MeasureResult instances. Validated against uncertainties module."""
    val1 = array([1.0, 2.0, 3.0])
    unc1 = array([0.1, 0.1, 0.1])
    val2 = array([2.0, 3.0, 4.0])
    unc2 = array([0.2, 0.2, 0.2])
    
    mr1 = MeasureResult(val1.copy(), unc1.copy(), unit="m")
    mr2 = MeasureResult(val2.copy(), unc2.copy(), unit="m")
    
    expected = uarray(val1, unc1) + uarray(val2, unc2)
    
    result = mr1 + mr2
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")

def test_sub_measure_results():
    """Test subtracting two MeasureResult instances. Validated against uncertainties module."""
    val1 = array([5.0, 6.0, 7.0])
    unc1 = array([0.1, 0.1, 0.1])
    val2 = array([2.0, 3.0, 4.0])
    unc2 = array([0.2, 0.2, 0.2])
    
    mr1 = MeasureResult(val1.copy(), unc1.copy(), unit="m")
    mr2 = MeasureResult(val2.copy(), unc2.copy(), unit="m")
    
    expected = uarray(val1, unc1) - uarray(val2, unc2)
    
    result = mr1 - mr2
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")

def test_mul_measure_results():
    """Test multiplying two MeasureResult instances. Validated against uncertainties module."""
    val1 = array([2.0, 3.0, 4.0])
    unc1 = array([0.1, 0.2, 0.3])
    val2 = array([5.0, 6.0, 7.0])
    unc2 = array([0.5, 0.6, 0.7])
    
    mr1 = MeasureResult(val1.copy(), unc1.copy(), unit="m")
    mr2 = MeasureResult(val2.copy(), unc2.copy(), unit="s")
    
    expected = uarray(val1, unc1) * uarray(val2, unc2)
    
    result = mr1 * mr2
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m*s")

def test_truediv_measure_results():
    """Test dividing two MeasureResult instances. Validated against uncertainties module."""
    val1 = array([10.0, 20.0, 30.0])
    unc1 = array([1.0, 2.0, 3.0])
    val2 = array([2.0, 4.0, 5.0])
    unc2 = array([0.1, 0.2, 0.25])
    
    mr1 = MeasureResult(val1.copy(), unc1.copy(), unit="m")
    mr2 = MeasureResult(val2.copy(), unc2.copy(), unit="s")
    
    expected = uarray(val1, unc1) / uarray(val2, unc2)
    
    result = mr1 / mr2
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m/s")

def test_mul_measure_results_unit_combination():
    """Test that multiplying creates correct unit combinations."""
    mr1 = MeasureResult(array([2.0]), array([0.1]), unit="m")
    mr2 = MeasureResult(array([3.0]), array([0.1]), unit="m")
    
    result = mr1 * mr2
    assert result.unit == Unit("m2")

def test_truediv_measure_results_unit_division():
    """Test that dividing creates correct unit divisions."""
    mr1 = MeasureResult(array([10.0]), array([1.0]), unit="m")
    mr2 = MeasureResult(array([2.0]), array([0.1]), unit="m")
    
    result = mr1 / mr2
    assert result.unit == dimensionless_unscaled

def test_unit_conversion_before_add():
    """Test adding MeasureResult with different but compatible units."""
    mr1 = MeasureResult(array([1000.0]), array([10.0]), unit="mm")
    mr2 = MeasureResult(array([1.0]), array([0.01]), unit="m")
    
    # Should convert mr2 to mm (1 m = 1000 mm)
    result = mr1 + mr2
    
    assert allclose(result.val, [2000.0])
    assert result.unit == Unit("mm")

def test_iadd_float():
    """Test in-place addition with float. Validated against uncertainties module."""
    val = array([1.0, 2.0, 3.0])
    unc = array([0.1, 0.1, 0.1])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) + 2.0
    
    original_id = id(mr)
    mr += 2.0
    
    # Should modify in place
    assert id(mr) == original_id
    assert allclose(mr.val, [e.nominal_value for e in expected])
    assert allclose(mr.unc, [e.std_dev for e in expected])

def test_isub_float():
    """Test in-place subtraction with float. Validated against uncertainties module."""
    val = array([5.0, 6.0, 7.0])
    unc = array([0.1, 0.1, 0.1])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) - 2.0
    
    mr -= 2.0
    
    assert allclose(mr.val, [e.nominal_value for e in expected])
    assert allclose(mr.unc, [e.std_dev for e in expected])

def test_imul_float():
    """Test in-place multiplication with float. Validated against uncertainties module."""
    val = array([2.0, 3.0, 4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) * 2.0
    
    mr *= 2.0
    
    assert allclose(mr.val, [e.nominal_value for e in expected])
    assert allclose(mr.unc, [e.std_dev for e in expected])

def test_itruediv_float():
    """Test in-place division with float. Validated against uncertainties module."""
    val = array([10.0, 20.0, 30.0])
    unc = array([1.0, 2.0, 3.0])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) / 2.0
    
    mr /= 2.0
    
    assert allclose(mr.val, [e.nominal_value for e in expected])
    assert allclose(mr.unc, [e.std_dev for e in expected])

def test_add_float_frozen():
    """Test adding float to FrozenMeasureResult. Validated against uncertainties module."""
    val = array([1.0, 2.0, 3.0])
    unc = array([0.1, 0.1, 0.1])
    fmr = FrozenMeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) + 5.0
    
    result = fmr + 5.0
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m")
    assert isinstance(result, FrozenMeasureResult)  # Result should be MeasureResult, not FrozenMeasureResult

def test_mul_quantity_frozen():
    """Test multiplying FrozenMeasureResult by Quantity. Validated against uncertainties module."""
    val = array([2.0, 3.0, 4.0])
    unc = array([0.1, 0.2, 0.3])
    fmr = FrozenMeasureResult(val.copy(), unc.copy(), unit="m")
    
    q = Quantity(5.0, "s")
    expected = uarray(val, unc) * q.value
    
    result = fmr * q
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m*s")

def test_mul_measure_results_frozen():
    """Test multiplying FrozenMeasureResult by MeasureResult. Validated against uncertainties module."""
    fmr = FrozenMeasureResult(array([2.0, 3.0]), array([0.1, 0.2]), unit="m")
    mr = MeasureResult(array([5.0, 6.0]), array([0.5, 0.6]), unit="s")
    
    expected = uarray(array([2.0, 3.0]), array([0.1, 0.2])) * uarray(array([5.0, 6.0]), array([0.5, 0.6]))
    
    result = fmr * mr
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
    assert result.unit == Unit("m*s")

def test_add_with_nan():
    """Test adding when some values are NaN."""
    val = array([1.0, nan, 3.0])
    unc = array([0.1, 0.1, 0.1])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    result = mr + 2.0
    
    assert allclose(result.val[0], 3.0)
    assert isnan(result.val[1])
    assert allclose(result.val[2], 5.0)
    assert allclose(result.unc[0], 0.1)
    assert allclose(result.unc[2], 0.1)

def test_mul_with_nan():
    """Test multiplying when some values are NaN."""
    val = array([2.0, nan, 4.0])
    unc = array([0.1, 0.2, 0.3])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    result = mr * 2.0
    
    assert allclose(result.val[0], 4.0)
    assert isnan(result.val[1])
    assert allclose(result.val[2], 8.0)

def test_wavelength_squared_unit():
    """Test that squaring wavelength gives correct unit."""
    mr = MeasureResult(array([100.0]), array([1.0]), unit="Angstrom")
    result = mr ** 2
    assert result.unit == Unit("Angstrom2")

def test_flux_times_wavelength():
    """Test flux * wavelength = strength."""
    val1 = array([1.0])
    unc1 = array([0.01])
    val2 = array([6563.0])
    unc2 = array([10.0])
    
    flux = MeasureResult(val1.copy(), unc1.copy(), unit="erg/(s*cm**2*Angstrom)")
    wavelength = MeasureResult(val2.copy(), unc2.copy(), unit="Angstrom")
    
    strength = flux * wavelength
    assert strength.unit == Unit("erg/(s*cm**2)")

def test_dimensionless_ratio():
    """Test creating a dimensionless ratio."""
    a = MeasureResult(array([5.0]), array([0.1]), unit="m")
    b = MeasureResult(array([2.0]), array([0.05]), unit="m")
    
    expected = uarray(array([5.0]), array([0.1])) / uarray(array([2.0]), array([0.05]))
    
    ratio = a / b
    assert ratio.unit == dimensionless_unscaled
    assert allclose(ratio.val, [e.nominal_value for e in expected])
    assert allclose(ratio.unc, [e.std_dev for e in expected])

def test_divide_by_zero_handling():
    """Test division by zero produces NaN/inf appropriately."""
    val = array([1.0, 2.0])
    unc = array([0.1, 0.2])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    divisor = MeasureResult(array([0.0, 0.0]), array([0.0, 0.0]), unit="s")
    
    result = mr / divisor
    
    # Result should be inf since we have finite/0
    assert not isfinite(result.val[0]) or isnan(result.val[0])

def test_zero_uncertainty():
    """Test operations with zero uncertainty."""
    val = array([5.0, 10.0])
    unc = array([0.0, 0.0])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) * 3.0
    
    result = mr * 3.0
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])

def test_large_uncertainty():
    """Test operations with large uncertainties."""
    val = array([1.0, 2.0])
    unc = array([10.0, 20.0])
    mr = MeasureResult(val.copy(), unc.copy(), unit="m")
    
    expected = uarray(val, unc) * 2.0
    
    result = mr * 2.0
    
    assert allclose(result.val, [e.nominal_value for e in expected])
    assert allclose(result.unc, [e.std_dev for e in expected])
