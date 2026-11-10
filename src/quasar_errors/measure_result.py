import math
from collections.abc import Callable
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    TypeVar,
    final,
    overload,
)

import numpy as np
from astropy.constants import c
from astropy.units import Quantity, Unit, dimensionless_unscaled
from numpy.typing import DTypeLike
from pydantic import (
    BaseModel,
    field_validator,
    model_serializer,
    model_validator,
)
from quasar_typing.astropy import CompositeUnitType, UnitType
from quasar_typing.numpy import BoolVector, FloatVector

F = TypeVar("F", bound=Callable)

@overload
def assert_is_not_logarithmic(func: None) -> Callable[[F], F]: ...

@overload
def assert_is_not_logarithmic[F](func: F) -> F: ...

@final
def assert_is_not_logarithmic[F](
    func: F | None,
) -> F | Callable[[F], F]:
    def wrapper(f: F) -> F:
        def inner(self, *args, **kwargs):
            if self.log_base is not None:
                raise ValueError(
                    f"Cannot call `{f.__name__}` on a logarithmic quantity."
                )
            return f(self, *args, **kwargs)
        return inner
    return wrapper if func is None else wrapper(func)

###

class _MeasureResult(BaseModel):
    val: FloatVector
    unc: FloatVector
    unit: UnitType | CompositeUnitType = dimensionless_unscaled
    log_base: float | None = None

    frozen: ClassVar[bool]

    @model_validator(mode="before")
    def remove_frozen_from_dict_input(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "frozen" in values:
                assert cls.frozen == values["frozen"]
            return {k: v for k, v in values.items() if k != "frozen"}
        return values

    @model_validator(mode="before")
    @classmethod
    def coerce_serialized_data(cls, values: Any) -> Any:
        if isinstance(values, dict) and all(
            k in values for k in ("val", "unc")
        ):
            f = lambda val: math.nan if val is None else float(val)
            values["val"] = np.fromiter(map(f, values["val"]), np.float64)
            values["unc"] = np.fromiter(map(f, values["unc"]), np.float64)
        return values
    
    @model_validator(mode="after")
    def check_array_lengths(self) -> Self:
        if self.val.size != self.unc.size:
            raise ValueError(
                f"`val` {self.val.shape} and `unc` {self.unc.shape} "
                "must have the same length"
            )
        return self


    @field_validator("val", "unc", mode="after")
    @classmethod
    def ensure_readonly_if_frozen(cls, v: FloatVector) -> FloatVector:
        v.flags.writeable = not getattr(cls, "frozen", False)
        return v


    @model_serializer(when_used="python")
    def serialize_model_to_python(self) -> dict[str, Any]:
        """Serialize the model, including the 'frozen' attribute."""
        return {
            "val": self.val,
            "unc": self.unc,
            "unit": self.unit,
            "log_base": self.log_base,
            "frozen": self.frozen,
        }
    

    @model_serializer(when_used="json")
    def serialize_model_to_json(self) -> dict[str, Any]:
        """Serialize the model, including the 'frozen' attribute."""
        mask = np.isfinite(self.val) & np.isfinite(self.unc)
        return {
            "val": [v if m else None for v, m in zip(self.val, mask)],
            "unc": [u if m else None for u, m in zip(self.unc, mask)],
            "unit": str(self.unit),
            "log_base": self.log_base,
            "frozen": self.frozen,
        }

    ###

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
            val=np.full(length, np.nan, dtype=np.float64),
            unc=np.full(length, 0.0, dtype=np.float64),
            unit=Unit(unit),
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

        return cls(val=val, unc=unc, unit=Unit(unit))

    def to_tuple(self) -> tuple[FloatVector, FloatVector]:
        return (self.val, self.unc)

    ### Copy

    @overload
    def copy(self, freeze: Literal[True]) -> "FrozenMeasureResult": ...
    
    @overload
    def copy(self, freeze: Literal[False] = False) -> "MeasureResult": ...

    @final
    def copy(self, freeze: bool = False) -> "FrozenMeasureResult | MeasureResult":
        cls = FrozenMeasureResult if freeze else MeasureResult
        return cls(
            val=self.val.copy(), 
            unc=self.unc.copy(), 
            unit=self.unit,
            log_base=self.log_base,
        )

    def freeze(self) -> "FrozenMeasureResult":
        return self if self.frozen else self.copy(freeze=True)

    def unfreeze(self) -> "MeasureResult":
        return self.copy(freeze=False)

    ### Data types

    def astype(self, dtype: DTypeLike) -> Self:
        return self.model_validate({
            "val": self.val.astype(dtype),
            "unc": self.unc.astype(dtype),
            "unit": self.unit,
            "log_base": self.log_base,
        })

    @property
    def dtype(self) -> DTypeLike:
        return self.val.dtype

    ### Units

    @assert_is_not_logarithmic
    def _set_unit(self, unit: UnitType | CompositeUnitType | Any) -> None:
        """Forcefully modify the 'unit' attribute."""
        object.__setattr__(self, "unit", Unit(unit))

    @assert_is_not_logarithmic
    def to(self, unit: UnitType | CompositeUnitType | Any) -> Self:
        unit = Unit(unit)
        factor = (1.0 * self.unit).to(unit).value
        return self.model_validate({
            "val": self.val * factor,
            "unc": self.unc * abs(factor),
            "unit": unit,
        })

    @assert_is_not_logarithmic
    def to_c(self) -> Self:
        """Convert a unitless MeasureResult ratio to a velocity in km/s."""
        out: MeasureResult = self.to(1) * c.to("km/s").value
        out._set_unit("km/s")
        return out.freeze() if isinstance(self, FrozenMeasureResult) else out

    ### Logarithmic Operations

    def log(self, base: float = math.e) -> Self:
        """
        Create a logarithmic version of this MeasureResult with the specified 
        base.
        """
        if self.log_base is None:
            val = np.log(self.val) / math.log(base)
            unc = self.unc / self.val / math.log(base)
        else:
            val = self.val * math.log(self.log_base) / math.log(base)
            unc = self.unc * math.log(self.log_base) / math.log(base)

        return self.model_validate({
            "val": val,
            "unc": unc,
            "unit": self.unit,
            "log_base": base,
        })

    def unlog(self) -> Self:
        """
        Convert a logarithmic MeasureResult back to its linear form. 
        
        If the MeasureResult is not logarithmic, it is returned as is.
        """
        if self.log_base is None:
            return self

        val = np.exp(self.val * math.log(self.log_base))
        unc = self.unc * val * math.log(self.log_base)

        return self.model_validate({
            "unc": unc,
            "val": val,
            "unit": self.unit,
            "log_base": None
        })

    ### Mathematical operations

    @assert_is_not_logarithmic
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

        return self.model_validate({
            "val": val,
            "unc": unc,
            "unit": self.unit,
        })

    @assert_is_not_logarithmic
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

        return self.model_validate({
            "val": val,
            "unc": unc,
            "unit": self.unit,
        })

    @assert_is_not_logarithmic
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

        return self.model_validate({
            "val": val,
            "unc": unc,
            "unit": unit,
        })

    @assert_is_not_logarithmic
    def __rmul__(self, num: float | Quantity | Self) -> Self:
        return self * num

    @assert_is_not_logarithmic
    def __truediv__(self, num: float | Quantity | Self) -> Self:
        if isinstance(num, (float, int, Quantity)):
            return self * (1 / num)

        val = np.full(self.val.size, np.nan, dtype=np.float64)
        unc = np.full(self.unc.size, 0.0, dtype=np.float64)
        mask = np.isfinite(self.val) & np.isfinite(num.val) & (num.val != 0)

        val[mask] = self.val[mask] / num.val[mask]
        unc[mask] = np.hypot(
            self.unc[mask] / num.val[mask],
            self.val[mask] * num.unc[mask] / num.val[mask]**2,
        )
        unit = self.unit / num.unit

        return self.model_validate({
            "val": val,
            "unc": unc,
            "unit": unit,
        })

    @assert_is_not_logarithmic
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

        return self.model_validate({
            "val": val,
            "unc": unc,
            "unit": unit,
        })

    @assert_is_not_logarithmic
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

        return self.model_validate({
            "val": val,
            "unc": unc,
            "unit": unit,
        })

    @assert_is_not_logarithmic
    def __abs__(self) -> Self:
        val = np.full(self.val.size, np.nan, dtype=np.float64)
        mask = np.isfinite(self.val)
        val[mask] = abs(self.val[mask])
        return self.model_validate({
            "val": val,
            "unc": self.unc.copy(),
            "unit": self.unit,
        })

class FrozenMeasureResult(_MeasureResult, frozen=True):
    frozen: ClassVar[Literal[True]] = True

class MeasureResult(_MeasureResult):
    frozen: ClassVar[Literal[False]] = False

    @assert_is_not_logarithmic
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

    @assert_is_not_logarithmic
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

    @assert_is_not_logarithmic
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

    @assert_is_not_logarithmic
    def remove_negatives(self) -> None:
        is_negative = self.val < 0
        self.val[is_negative] = 0
        self.unc[is_negative] = np.inf

    ### Arithmetic

    @assert_is_not_logarithmic
    def __iadd__(self, other: float | Quantity | Self) -> Self:
        """
        Inplace addition.
        """
        temp = self + other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self

    @assert_is_not_logarithmic
    def __isub__(self, other: float | Quantity | Self) -> Self:
        """
        Inplace subtraction.
        """
        temp = self - other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self

    @assert_is_not_logarithmic
    def __imul__(self, other: float | Quantity | Self) -> Self:
        """
        Inplace multiplication.
        """
        temp = self * other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self

    @assert_is_not_logarithmic
    def __itruediv__(self, other: float | Quantity | Self) -> Self:
        """
        Inplace true division.
        """
        temp = self / other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self

    @assert_is_not_logarithmic
    def __ipow__(self, other: float) -> Self:
        temp = self ** other
        self.val[:] = temp.val
        self.unc[:] = temp.unc
        return self


class MeasureResultDict(BaseModel):
    """
    Dictionary-like container for measure results.
    """
    results: dict[str, MeasureResult | FrozenMeasureResult]

    @classmethod
    def fromMeasureResultDict(
        cls, 
        results: dict[str, MeasureResult | FrozenMeasureResult],
    ) -> Self:
        return cls(results=results)

    def toMeasureResultDict(self) -> dict[str, MeasureResult | FrozenMeasureResult]:
        return self.results