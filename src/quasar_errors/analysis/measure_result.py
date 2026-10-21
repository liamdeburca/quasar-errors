from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pydantic
from astropy.constants import c
from astropy.units import Quantity, Unit, dimensionless_unscaled
from quasar_typing.astropy import CompositeUnitType, UnitType
from quasar_typing.numpy import BoolVector, FloatVector


class _MeasureResult(ABC):
    @abstractmethod
    def __init__(
        self, 
        val: FloatVector, 
        unc: FloatVector, 
        unit: UnitType | CompositeUnitType = dimensionless_unscaled,
    ) -> None:
        ...


    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += ", ".join(
            f"{v:.1f}±{u:.1f}" if np.isfinite(v) else "nan"
            for v, u in zip(self.val, self.unc)
        )
        return s + ")"


    @classmethod
    def create_empty(
        cls,
        length: int,
        unit: UnitType | CompositeUnitType = dimensionless_unscaled,
    ) -> Self:
        if unit is None:
            unit = dimensionless_unscaled

        return cls(
            np.full(length, np.nan, dtype=np.float64),
            np.full(length, 0.0, dtype=np.float64),
            unit = Unit(unit)
        )
        
    @classmethod
    def from_list_of_tuples(
        cls,
        tuple_list: list[tuple[float, float]],
        *,
        length: int | None = None,
        unit: UnitType | CompositeUnitType = dimensionless_unscaled,
    ) -> Self:
        n_tups = len(tuple_list)
        length = length or len(tuple_list)

        if length > n_tups:
            val = np.full(length, np.nan, dtype=np.float64)
            unc = np.full(length, 0.0, dtype=np.float64)

            val[:n_tups] = [tup[0] for tup in tuple_list]
            unc[:n_tups] = [tup[1] for tup in tuple_list]
        else:
            val = np.fromiter(
                (tup[0] for tup in tuple_list), 
                dtype=np.float64, 
                count=length,
            )
            unc = np.fromiter(
                (tup[1] for tup in tuple_list), 
                dtype=np.float64, 
                count=length,
            )

        return cls(val, unc, unit=Unit(unit))

    def to_tuple(self) -> tuple[FloatVector, FloatVector]:
        return (self.val, self.unc)

    def copy(self, freeze: bool = False) -> "FrozenMeasureResult | MeasureResult":
        cls = FrozenMeasureResult if freeze else MeasureResult
        return cls(self.val.copy(), self.unc.copy(), unit=self.unit)

    def freeze(self) -> "FrozenMeasureResult":
        if isinstance(self, FrozenMeasureResult):
            return self
        return self.copy(freeze=True)

    def unfreeze(self) -> "MeasureResult":
        return self.copy(freeze=False)

    def _set_unit(self, unit: UnitType | CompositeUnitType | Any) -> None:
        """Forcefully modify the 'unit' attribute."""
        object.__setattr__(self, "unit", Unit(unit))

    def to(self, unit: UnitType | CompositeUnitType | Any) -> Self:
        unit = Unit(unit)
        factor = (1.0 * self.unit).to(unit).value
        return self.__class__(
            self.val * factor,
            self.unc * abs(factor),
            unit=unit,
        )

    def to_c(self) -> Self:
        """Convert a unitless MeasureResult ratio to a velocity in km/s."""
        out = self.to(1)
        out: MeasureResult = self.to(1) * c.to("km/s").value
        out._set_unit("km/s")
        return out.freeze() if isinstance(self, FrozenMeasureResult) else out
        
    def __add__(self, num: float | Quantity | Self) -> Self:
        """
        Add a number or another MeasureResult to this MeasureResult. The unit of 
        the result will be identical to the unit of the left MeasureResult.
        """
        val = np.full(self.val.size, np.nan, dtype=np.float64)
        unc = np.full(self.unc.size, 0.0, dtype=np.float64)
        mask = np.isfinite(self.val)

        if isinstance(num, (float, int)):
            val[mask] = self.val[mask] + num
            unc[mask] = self.unc[mask]
        elif isinstance(num, Quantity):
            num = num.to(self.unit)
            val[mask] = self.val[mask] + num.value
            unc[mask] = self.unc[mask]
        else:
            mask &= np.isfinite(num.val)
            num = num.to(self.unit)
            val[mask] = self.val[mask] + num.val[mask]
            unc[mask] = np.hypot(self.unc[mask], num.unc[mask])

        return self.__class__(val, unc, unit=self.unit)  

    def __sub__(self, num: float | Quantity | Self) -> Self:
        """
        Subtraction.
        """
        val = np.full(self.val.size, np.nan, dtype=np.float64)
        unc = np.full(self.unc.size, 0.0, dtype=np.float64)
        mask = np.isfinite(self.val)

        if isinstance(num, (float, int)):
            val[mask] = self.val[mask] - num
            unc[mask] = self.unc[mask]
        elif isinstance(num, Quantity):
            num = num.to(self.unit)
            val[mask] = self.val[mask] - num.value
            unc[mask] = self.unc[mask]
        else:
            mask &= np.isfinite(num.val)
            num = num.to(self.unit)
            val[mask] = self.val[mask] - num.val[mask]
            unc[mask] = np.hypot(self.unc[mask], num.unc[mask])

        return self.__class__(val, unc, unit=self.unit)
    
    def __mul__(self, num: float | Quantity | Self) -> Self:
        """
        Multiplication.
        """
        val = np.full(self.val.size, np.nan, dtype=np.float64)
        unc = np.full(self.unc.size, 0.0, dtype=np.float64)
        mask = np.isfinite(self.val)

        if isinstance(num, (float, int)):
            val[mask] = self.val[mask] * num
            unc[mask] = self.unc[mask] * abs(num)
            unit = self.unit
        elif isinstance(num, Quantity):
            val[mask] = self.val[mask] * num.value
            unc[mask] = self.unc[mask] * abs(num.value)
            unit = self.unit * num.unit
        else:
            mask &= np.isfinite(num.val)
            val[mask] = self.val[mask] * num.val[mask]
            unc[mask] = np.hypot(
                num.val[mask] * self.unc[mask],
                num.unc[mask] * self.val[mask],
            )
            unit = self.unit * num.unit

        return self.__class__(val, unc, unit=unit)

    def __rmul__(self, num: float | Quantity | Self) -> Self:
        return self * num

    def __truediv__(self, num: float | Quantity | Self) -> Self:
        if isinstance(num, (float, int, Quantity)):
            return self * (1 / num)

        val = np.full(self.val.size, np.nan, dtype=np.float64)
        unc = np.full(self.unc.size, 0.0, dtype=np.float64)
        mask = np.isfinite(self.val) & np.isfinite(num.val)

        val[mask] = self.val[mask] / num.val[mask]
        unc[mask] = np.hypot(
            self.unc[mask] / num.val[mask],
            self.val[mask] * num.unc[mask] / num.val[mask]**2,
        )
        unit = self.unit / num.unit

        return self.__class__(val, unc, unit=unit)

    def __rtruediv__(self, num: float | Quantity | Self) -> Self:
        if isinstance(num, (float, int, Quantity)):
            return num * (self ** -1)
        
        val = np.full(self.val.size, np.nan, dtype=np.float64)
        unc = np.full(self.unc.size, 0.0, dtype=np.float64)
        mask = np.isfinite(self.val) & np.isfinite(num.val)

        val[mask] = num.val[mask] / self.val[mask]
        unc[mask] = np.hypot(
            num.unc[mask] / self.val[mask],
            num.val[mask] * self.unc[mask] / self.val[mask]**2,
        )
        unit = num.unit / self.unit

        return self.__class__(val, unc, unit=unit)

    def __pow__(self, num: float) -> Self:
        val = np.full(self.val.size, np.nan, dtype=np.float64)
        unc = np.full(self.unc.size, 0.0, dtype=np.float64)
        mask = np.isfinite(self.val)

        if num == 0:
            val[mask] = 1.0
            unc[mask] = 0.0
            unit = dimensionless_unscaled
        else:
            val[mask] = self.val[mask] ** num
            unc[mask] = abs(num * self.val[mask] ** (num - 1) * self.unc[mask])
            unit = self.unit ** num

        return self.__class__(val, unc, unit=unit)

    def __abs__(self) -> Self:
        val = np.full(self.val.size, np.nan, dtype=np.float64)
        mask = np.isfinite(self.val)
        val[mask] = abs(self.val[mask])
        return self.__class__(val, self.unc.copy(), unit=self.unit)

@pydantic.dataclasses.dataclass(frozen=True)
class FrozenMeasureResult(_MeasureResult):
    val: FloatVector
    unc: FloatVector
    unit: UnitType | CompositeUnitType = dimensionless_unscaled

@pydantic.dataclasses.dataclass
class MeasureResult(_MeasureResult):
    val: FloatVector
    unc: FloatVector
    unit: UnitType | CompositeUnitType = dimensionless_unscaled

    def update(
        self,
        new_values: Self,
        index: int | slice | BoolVector | None = None,
    ) -> None:
        """
        Modifies this instance's 'val' and 'unc' arrays, updating them with the
        values from the provided 'new_values' instance at the specified
        index/indices.
        """
        if index is None:
            index = slice(0, min(self.val.shape[0], new_values.val.shape[0]))

        self.val[index] = new_values.val[index]
        self.unc[index] = new_values.unc[index]

    def update_with_single_values(
        self,
        new_values: list[float],
        index: slice | BoolVector | None = None,
    ) -> None:
        """
        Modifies this instance's 'val' array, adding new values at the specified
        index/indices. The 'unc' array is set to 0 for these indices, indicating
        no valid uncertainty for these values.
        """
        if index is None:
            index = slice(0, min(self.val.shape[0], len(new_values)))

        if isinstance(index, slice):
            new_values = new_values[index]
        else:
            new_values = [val for i, val in enumerate(new_values) if index[i]]

        self.val[index] = new_values
        self.unc[index] = 0

    def update_with_list_of_tuples(
        self,
        new_values: list[tuple[float, float]],
        index: slice | BoolVector | None = None,
    ) -> None:
        """
        Modifies this instance's 'val' and 'unc' arrays, adding new values at the
        specified index/indices. The new values are provided as a list of tuples,
        where each tuple contains a (value, uncertainty) pair.
        """
        if index is None:
            index = slice(0, min(self.val.shape[0], len(new_values)))

        if isinstance(index, slice):
            new_values = new_values[index]
        else:
            new_values = [tup for i, tup in enumerate(new_values) if index[i]]

        self.val[index] = [tup[0] for tup in new_values]
        self.unc[index] = [tup[1] for tup in new_values]

    def remove_negatives(self) -> None:
        is_negative = self.val < 0
        self.val[is_negative] = 0
        self.unc[is_negative] = np.inf

    ### Arithmetic

    def __iadd__(self, other: float | Quantity | Self) -> Self:
        """
        Inplace addition.
        """
        temp = self + other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self

    def __isub__(self, other: float | Quantity | Self) -> Self:
        """
        Inplace subtraction.
        """
        temp = self - other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self

    def __imul__(self, other: float | Quantity | Self) -> Self:
        """
        Inplace multiplication.
        """
        temp = self * other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self

    def __itruediv__(self, other: float | Quantity | Self) -> Self:
        """
        Inplace true division.
        """
        temp = self / other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self

    def __ipow__(self, other: float) -> Self:
        temp = self ** other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self
