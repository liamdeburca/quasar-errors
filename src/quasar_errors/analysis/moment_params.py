from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

from quasar_typing.numpy import BoolMatrix, FloatMatrix
from quasar_typing.string_selections import BackgroundFlux

from .measure_result import FrozenMeasureResult


@dataclass(frozen=True)
class MomentParams:
    analyser: object
    from_model: bool = field(
        default=False,
        kw_only=True,
    )
    bg_flux: BackgroundFlux = field(
        default_factory=lambda: BackgroundFlux({"all", "em"}), 
        kw_only=True,
    )
    centre: Literal["wave", "centroid", "median", "mode"] = field(
        default="wave", 
        kw_only=True
    )

    @cached_property
    def tot(self) -> FrozenMeasureResult:
        return self.analyser._measureZerothMoment(
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )

    @cached_property
    def mask(self) -> BoolMatrix:
        return self.analyser._getMask(
            centre=self.centre,
            from_model=self.from_model,
            simplify=False,
        )

    @cached_property
    def xs(self) -> FloatMatrix:
        return self.analyser.x[None, :]

    @cached_property
    def ys(self) -> FloatMatrix:
        return self.analyser._getFlux(
            bg_flux=self.bg_flux,
            ensure_not_negative=False,
            simplify=False,
        )

    @cached_property
    def dxs(self) -> FloatMatrix:
        return self.analyser.dx[None, :]

    @cached_property
    def dys(self) -> FloatMatrix:
        return self.analyser.dy[None, :]

    @cached_property
    def m1(self) -> FrozenMeasureResult:
        return self.analyser._measureFirstMoment(
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )

    @cached_property
    def cm2(self) -> FrozenMeasureResult:
        return self.analyser._measureSecondCentralMoment(
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )

    @cached_property
    def cm3(self) -> FrozenMeasureResult:
        return self.analyser._measureThirdCentralMoment(
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )

    @cached_property
    def cm4(self) -> FrozenMeasureResult:
        return self.analyser._measureFourthCentralMoment(
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )
