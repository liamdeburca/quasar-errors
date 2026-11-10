"""
Test suite for logarithmic operations on MeasureResult and FrozenMeasureResult.

Tests cover:
- log(base) and unlog() round-trips
- Uncertainty propagation through logarithmic transformations
- Operations blocked on logarithmic quantities
- Serialization preserves log_base
- Frozen vs mutable behavior
"""

import math

import pytest
from astropy.units import Unit, dimensionless_unscaled
from numpy import allclose, array, isfinite, isnan, nan

from quasar_errors.measure_result import (
    FrozenMeasureResult,
    MeasureResult,
)


class TestLogUnlogRoundTrip:
    """Test that log() and unlog() correctly invert each other."""

    def test_log_base10_unlog_round_trip(self):
        """Test log(base=10) -> unlog() returns original values."""
        val = array([1.0, 10.0, 100.0, 1000.0])
        unc = array([0.1, 1.0, 10.0, 100.0])
        mr = MeasureResult(val=val.copy(), unc=unc.copy(), unit="erg/s")

        logged = mr.log(10)
        restored = logged.unlog()

        assert allclose(restored.val, mr.val, rtol=1e-10)
        assert allclose(restored.unc, mr.unc, rtol=1e-10)
        assert restored.unit == mr.unit
        assert restored.log_base is None

    def test_log_base2_unlog_round_trip(self):
        """Test log(base=2) -> unlog() returns original values."""
        val = array([1.0, 2.0, 4.0, 8.0, 16.0])
        unc = array([0.05, 0.1, 0.2, 0.4, 0.8])
        mr = MeasureResult(val=val.copy(), unc=unc.copy(), unit="km/s")

        logged = mr.log(2)
        restored = logged.unlog()

        assert allclose(restored.val, mr.val, rtol=1e-10)
        assert allclose(restored.unc, mr.unc, rtol=1e-10)
        assert restored.unit == mr.unit
        assert restored.log_base is None

    def test_log_base_e_unlog_round_trip(self):
        """Test log(base=e) -> unlog() returns original values."""
        val = array([math.e, math.e**2, math.e**3])
        unc = array([0.1, 0.2, 0.3])
        mr = MeasureResult(val=val.copy(), unc=unc.copy(), unit="mag")

        # Use natural log base approximately
        logged = mr.log(int(math.e))
        restored = logged.unlog()

        assert allclose(restored.val, mr.val, rtol=1e-9)
        assert allclose(restored.unc, mr.unc, rtol=1e-9)
        assert restored.unit == mr.unit
        assert restored.log_base is None

    def test_unlog_on_linear_quantity_is_idempotent(self):
        """Test that unlog() on non-logged quantity returns self."""
        val = array([1.0, 2.0, 3.0])
        unc = array([0.1, 0.2, 0.3])
        mr = MeasureResult(val=val.copy(), unc=unc.copy(), unit="m")

        result = mr.unlog()

        # Should return the same instance or equivalent
        assert result is mr

    def test_frozen_log_unlog_round_trip(self):
        """Test log/unlog preserves frozen status."""
        val = array([10.0, 100.0, 1000.0])
        unc = array([1.0, 10.0, 100.0])
        fmr = FrozenMeasureResult(val=val.copy(), unc=unc.copy(), unit="erg/s")

        logged = fmr.log(10)
        assert isinstance(logged, FrozenMeasureResult)
        assert logged.log_base == 10
        assert not logged.val.flags.writeable

        restored = logged.unlog()
        assert isinstance(restored, FrozenMeasureResult)
        assert restored.log_base is None
        assert not restored.val.flags.writeable

    def test_mutable_log_unlog_round_trip(self):
        """Test log/unlog preserves mutable status."""
        val = array([10.0, 100.0, 1000.0])
        unc = array([1.0, 10.0, 100.0])
        mr = MeasureResult(val=val.copy(), unc=unc.copy(), unit="erg/s")

        logged = mr.log(10)
        assert isinstance(logged, MeasureResult)
        assert logged.log_base == 10
        assert logged.val.flags.writeable

        restored = logged.unlog()
        assert isinstance(restored, MeasureResult)
        assert restored.log_base is None
        assert restored.val.flags.writeable


class TestLogarithmicUncertaintyPropagation:
    """Test that uncertainty propagation follows correct calculus."""

    def test_log_uncertainty_propagation_small_unc(self):
        """
        Test uncertainty propagation for logarithm: du' = du / (u * ln(b))
        For small relative uncertainties, log(u ± du) ≈ log(u) ± du/(u*ln(b))
        """
        val = array([100.0])
        unc = array([1.0])
        mr = MeasureResult(val=val.copy(), unc=unc.copy(), unit="erg/s")

        base = 10
        logged = mr.log(base)

        # Expected: val' = log10(100) = 2, unc' = 1 / (100 * ln(10))
        expected_val = math.log(100) / math.log(base)
        expected_unc = 1.0 / (100.0 * math.log(base))

        assert allclose(logged.val, [expected_val], rtol=1e-10)
        assert allclose(logged.unc, [expected_unc], rtol=1e-10)

    def test_log_uncertainty_propagation_large_unc(self):
        """
        Test logarithmic uncertainty with larger relative uncertainty.
        """
        val = array([10.0])
        unc = array([2.0])  # 20% relative uncertainty
        mr = MeasureResult(val=val.copy(), unc=unc.copy(), unit="mag")

        base = 10
        logged = mr.log(base)

        expected_val = math.log(10) / math.log(base)
        expected_unc = 2.0 / (10.0 * math.log(base))

        assert allclose(logged.val, [expected_val], rtol=1e-10)
        assert allclose(logged.unc, [expected_unc], rtol=1e-10)

    def test_base_conversion_different_bases(self):
        """
        Test converting already-logged values to different base.
        If log_base = b1 and we call log(b2), it converts:
        val_new = val_old * ln(b1) / ln(b2)
        unc_new = unc_old * ln(b1) / ln(b2)
        """
        val = array([2.0])  # log10(100)
        unc = array([0.1])
        mr = MeasureResult(val=val.copy(), unc=unc.copy(), unit="erg/s", log_base=10)

        # Convert from base 10 to base 2
        converted = mr.log(2)

        ln10 = math.log(10)
        ln2 = math.log(2)
        expected_val = 2.0 * ln10 / ln2
        expected_unc = 0.1 * ln10 / ln2

        assert allclose(converted.val, [expected_val], rtol=1e-10)
        assert allclose(converted.unc, [expected_unc], rtol=1e-10)
        assert converted.log_base == 2


class TestLogBlocksArithmetic:
    """Test that @assert_is_not_logarithmic decorator blocks operations."""

    def test_add_blocked_on_logged_quantity(self):
        """Cannot add to logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged + 5.0

    def test_sub_blocked_on_logged_quantity(self):
        """Cannot subtract from logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged - 1.0

    def test_mul_blocked_on_logged_quantity(self):
        """Cannot multiply logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged * 2.0

    def test_truediv_blocked_on_logged_quantity(self):
        """Cannot divide logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged / 2.0

    def test_pow_blocked_on_logged_quantity(self):
        """Cannot raise logarithmic quantity to power."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged ** 2

    def test_abs_blocked_on_logged_quantity(self):
        """Cannot take absolute value of logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            abs(logged)

    def test_iadd_blocked_on_logged_quantity(self):
        """Cannot do in-place addition on logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged += 1.0

    def test_isub_blocked_on_logged_quantity(self):
        """Cannot do in-place subtraction on logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged -= 1.0

    def test_imul_blocked_on_logged_quantity(self):
        """Cannot do in-place multiplication on logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged *= 2.0

    def test_itruediv_blocked_on_logged_quantity(self):
        """Cannot do in-place division on logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged /= 2.0

    def test_ipow_blocked_on_logged_quantity(self):
        """Cannot do in-place power on logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged **= 2

    def test_to_blocked_on_logged_quantity(self):
        """Cannot convert units of logarithmic quantity."""
        mr = MeasureResult(val=array([2.0, 3.0]), unc=array([0.1, 0.2]), unit="erg/s")
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged.to("erg/(s*cm**2)")

    def test_to_c_blocked_on_logged_quantity(self):
        """Cannot convert to velocity for logarithmic quantity."""
        mr = MeasureResult(
            val=array([0.1, 0.2]), unc=array([0.01, 0.02]), unit=dimensionless_unscaled
        )
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged.to_c()

    def test_update_blocked_on_logged_quantity(self):
        """Cannot update values on logarithmic quantity."""
        mr = MeasureResult(
            val=array([2.0, 3.0, 4.0]), unc=array([0.1, 0.2, 0.3]), unit="erg/s"
        )
        logged = mr.log(10)
        new_vals = MeasureResult(
            val=array([5.0, 6.0]), unc=array([0.5, 0.6]), unit="erg/s"
        )

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged.update(new_vals)

    def test_remove_negatives_blocked_on_logged_quantity(self):
        """Cannot remove negatives from logarithmic quantity."""
        mr = MeasureResult(
            val=array([-2.0, 3.0, -4.0]), unc=array([0.1, 0.2, 0.3]), unit="erg/s"
        )
        logged = mr.log(10)

        with pytest.raises(ValueError, match="Cannot call.*on a logarithmic"):
            logged.remove_negatives()


class TestLogSerializationPreservation:
    """Test that log_base is correctly preserved in serialization."""

    def test_python_serialization_includes_log_base(self):
        """Python serializer includes log_base in output."""
        mr = MeasureResult(val=array([1.0, 10.0]), unc=array([0.1, 1.0]), unit="erg/s")
        logged = mr.log(10)

        serialized = logged.model_dump(mode="python")

        assert "log_base" in serialized
        assert serialized["log_base"] == 10

    def test_json_serialization_includes_log_base(self):
        """JSON serializer includes log_base in output."""
        mr = MeasureResult(val=array([1.0, 10.0]), unc=array([0.1, 1.0]), unit="erg/s")
        logged = mr.log(10)

        serialized = logged.model_dump(mode="json")

        assert "log_base" in serialized
        assert serialized["log_base"] == 10

    def test_deserialize_logged_quantity_preserves_log_base(self):
        """Deserializing a logged quantity preserves log_base."""
        mr = MeasureResult(val=array([1.0, 10.0]), unc=array([0.1, 1.0]), unit="erg/s")
        logged = mr.log(10)

        serialized = logged.model_dump(mode="python")
        restored = MeasureResult.model_validate(serialized)

        assert restored.log_base == 10
        assert allclose(restored.val, logged.val)
        assert allclose(restored.unc, logged.unc)

    def test_json_serialize_with_nan_and_log_base(self):
        """JSON serialization handles NaN with log_base."""
        mr = MeasureResult(
            val=array([1.0, nan, 100.0]), unc=array([0.1, nan, 10.0]), unit="erg/s"
        )
        logged = mr.log(10)

        serialized = logged.model_dump(mode="json")

        assert serialized["log_base"] == 10
        assert serialized["val"][0] is not None
        assert serialized["val"][1] is None  # NaN becomes None
        assert serialized["val"][2] is not None


class TestLogEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_log_of_unity(self):
        """Test log(1) = 0 for any base."""
        mr = MeasureResult(val=array([1.0, 1.0]), unc=array([0.01, 0.02]))
        logged = mr.log(10)

        # log10(1) = 0
        assert allclose(logged.val, [0.0, 0.0], atol=1e-15)

    def test_log_with_zero_uncertainty(self):
        """Test logging value with zero uncertainty."""
        mr = MeasureResult(val=array([100.0]), unc=array([0.0]), unit="erg/s")
        logged = mr.log(10)

        expected_val = math.log(100) / math.log(10)
        expected_unc = 0.0 / (100.0 * math.log(10))

        assert allclose(logged.val, [expected_val])
        assert allclose(logged.unc, [expected_unc])

    def test_log_with_very_small_value(self):
        """Test logging very small value."""
        mr = MeasureResult(val=array([1e-10]), unc=array([1e-11]), unit="erg/s")
        logged = mr.log(10)

        expected_val = math.log(1e-10) / math.log(10)
        expected_unc = 1e-11 / (1e-10 * math.log(10))

        assert allclose(logged.val, [expected_val], rtol=1e-10)
        assert allclose(logged.unc, [expected_unc], rtol=1e-10)

    def test_log_with_very_large_value(self):
        """Test logging very large value."""
        mr = MeasureResult(val=array([1e30]), unc=array([1e29]), unit="erg/s")
        logged = mr.log(10)

        expected_val = math.log(1e30) / math.log(10)
        expected_unc = 1e29 / (1e30 * math.log(10))

        assert allclose(logged.val, [expected_val], rtol=1e-10)
        assert allclose(logged.unc, [expected_unc], rtol=1e-10)

    def test_unlog_with_nan(self):
        """Test unlog with NaN values."""
        mr = MeasureResult(
            val=array([2.0, nan, 3.0]), unc=array([0.1, nan, 0.2]), unit="erg/s", log_base=10
        )

        restored = mr.unlog()

        assert isfinite(restored.val[0])
        assert isnan(restored.val[1])
        assert isfinite(restored.val[2])

    def test_unit_preserved_through_log_unlog(self):
        """Test that unit is preserved through log/unlog cycle."""
        units = ["m", "erg/(s*cm**2)", "km/s", "mag", "Angstrom"]

        for unit in units:
            mr = MeasureResult(val=array([10.0]), unc=array([1.0]), unit=unit)
            logged = mr.log(10)
            restored = logged.unlog()

            assert restored.unit == Unit(unit)

    def test_large_array_log_unlog(self):
        """Test log/unlog on large array."""
        n = 10000
        val = array([10.0 ** (i % 20) for i in range(n)])
        unc = val * 0.1
        mr = MeasureResult(val=val.copy(), unc=unc.copy(), unit="erg/s")

        logged = mr.log(10)
        restored = logged.unlog()

        assert allclose(restored.val, mr.val, rtol=1e-9)
        assert allclose(restored.unc, mr.unc, rtol=1e-9)

    def test_copy_preserves_log_base(self):
        """Test that copy() preserves log_base."""
        mr = MeasureResult(val=array([1.0, 10.0]), unc=array([0.1, 1.0]), unit="erg/s")
        logged = mr.log(10)

        copied = logged.copy()

        assert copied.log_base == 10
        assert allclose(copied.val, logged.val)
        assert allclose(copied.unc, logged.unc)

    def test_freeze_preserves_log_base(self):
        """Test that freeze() preserves log_base."""
        mr = MeasureResult(val=array([1.0, 10.0]), unc=array([0.1, 1.0]), unit="erg/s")
        logged = mr.log(10)

        frozen = logged.freeze()

        assert frozen.log_base == 10
        assert isinstance(frozen, FrozenMeasureResult)
        assert allclose(frozen.val, logged.val)
