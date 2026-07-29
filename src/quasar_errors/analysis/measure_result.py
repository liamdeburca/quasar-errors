from typing import Self

from numpy import (
    float64,
    full,
    full_like,
    hypot,
    inf,
    isfinite,
    log,
    nan,
    zeros_like,
)
from pydantic.dataclasses import dataclass
from quasar_typing.numpy import BoolVector, FloatVector


@dataclass(repr=False)
class MeasureResult:
    val: FloatVector
    unc: FloatVector | None = None

    def __post_init__(self):
        if self.unc is None:
            self.unc = zeros_like(self.val)

    def __repr__(self) -> str:
        s = "MeasureResult("
        s += ", ".join(
            f"{v:.1f}±{u:.1f}" if isfinite(v) else "nan"
            for v, u in zip(self.val, self.unc)
        )
        s += ")"
        return s

    @classmethod
    def create_empty(
        cls,
        length: int,
    ) -> Self:
        return MeasureResult(full(length, nan, dtype=float64))

    @classmethod
    def from_list_of_tuples(
        cls,
        tuple_list: list[tuple[float, float]],
        *,
        length: int | None = None,
    ) -> Self:
        length = length or len(tuple_list)
        res = cls.create_empty(length)
        res.update_with_list_of_tuples(tuple_list, index=slice(0, length))
        return res

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
        self.unc[is_negative] = inf

    def to_tuple(self) -> tuple[FloatVector, FloatVector]:
        return (self.val, self.unc)

    ### Arithmetic

    def copy(self) -> Self:
        return MeasureResult(
            self.val.copy(),
            unc=self.unc.copy(),
        )

    ## Addition

    def __add__(self, num: float) -> Self:
        copy = self.copy()
        copy += num
        return copy

    def __iadd__(self, other: float | Self) -> Self:
        """
        Inplace addition.
        """
        if isinstance(other, MeasureResult):
            mask = isfinite(self.val) & isfinite(other.val)
            num1 = self.val[mask]
            d_num1 = self.unc[mask]

            num2 = other.val[mask]
            d_num2 = other.unc[mask]

            new_val = full_like(self.val, nan)
            new_unc = full_like(self.unc, 0)
            new_val[mask] = num1 + num2
            new_unc[mask] = hypot(d_num1, d_num2)

            self.val[:] = new_val
            self.unc[:] = new_unc
        else:
            self.val += other
        return self

    ## Subtraction

    def __sub__(self, num: float) -> Self:
        """
        Subtraction.
        """
        copy = self.copy()
        copy -= num
        return copy

    def __isub__(self, other: float | Self) -> Self:
        """
        Inplace subtraction.
        """
        if isinstance(other, MeasureResult):
            mask = isfinite(self.val) & isfinite(other.val)
            num1 = self.val[mask]
            d_num1 = self.unc[mask]

            num2 = other.val[mask]
            d_num2 = other.unc[mask]

            new_val = full_like(self.val, nan)
            new_unc = full_like(self.unc, 0)
            new_val[mask] = num1 - num2
            new_unc[mask] = hypot(d_num1, d_num2)

            self.val[:] = new_val
            self.unc[:] = new_unc
        else:
            self.val -= other
        return self

    ## Multiplication

    def __mul__(self, num: float | Self) -> Self:
        """
        Multiplication.
        """
        copy = self.copy()
        copy *= num
        return copy

    def __rmul__(self, num: float | Self) -> Self:
        return self * num

    def __imul__(self, other: float | Self) -> Self:
        """
        Inplace multiplication.
        """
        if isinstance(other, MeasureResult):
            mask = isfinite(self.val) & isfinite(other.val)
            num1 = self.val[mask]
            d_num1 = self.unc[mask]

            num2 = other.val[mask]
            d_num2 = other.unc[mask]

            new_val = full_like(self.val, nan)
            new_unc = full_like(self.unc, 0)
            new_val[mask] = num1 * num2
            new_unc[mask] = hypot(num2 * d_num1, num1 * d_num2)

            self.val[:] = new_val
            self.unc[:] = new_unc
        else:
            self.val *= other
            self.unc *= abs(other)
        return self

    ## Division

    def __truediv__(self, other: float | Self) -> Self:
        copy = self.copy()
        copy /= other
        return copy

    def __rtruediv__(self, other: float | Self) -> Self:
        copy = self.copy() ** -1
        copy *= other
        return copy

    def __itruediv__(self, other: float | Self) -> Self:
        if isinstance(other, MeasureResult):
            mask = isfinite(self.val) & isfinite(other.val) & (other.val != 0)
            num = self.val[mask]
            d_num = self.unc[mask]

            den = other.val[mask]
            d_den = other.unc[mask]

            new_val = full_like(self.val, nan)
            new_unc = full_like(self.unc, 0)
            new_val[mask] = num / den
            new_unc[mask] = hypot(d_num / den, num / den**2 * d_den)

            self.val[:] = new_val
            self.unc[:] = new_unc
        else:
            self.val /= other
            self.unc /= abs(other)
        return self

    ## Power

    def __pow__(self, num: float) -> Self:
        copy = self.copy()
        copy **= num
        return copy

    def __ipow__(self, other: float | Self) -> Self:
        if isinstance(other, MeasureResult):
            mask = isfinite(self.val) & isfinite(other.val) & (self.val > 0)

            new_val = full_like(self.val, nan)
            new_unc = full_like(self.unc, 0)

            new_val[mask] = self.val[mask] ** other.val[mask]
            new_unc[mask] = abs(new_val[mask]) * hypot(
                other.unc[mask] * log(self.val[mask]),
                other.val[mask] * self.unc[mask] / self.val[mask],
            )
            self.val[:] = new_val
            self.unc[:] = new_unc
        else:
            mask = isfinite(self.val) & (self.val > 0)

            new_val = full_like(self.val, nan)
            new_unc = full_like(self.unc, 0)

            new_val[mask] = self.val[mask] ** other
            new_unc[mask] = abs(
                other * self.val[mask] ** (other - 1) * self.unc[mask]
            )

            self.val[:] = new_val
            self.unc[:] = new_unc
        return self

    ## Absolute value

    def __abs__(self) -> Self:
        copy = self.copy()
        copy.val = abs(copy.val)
        return copy
