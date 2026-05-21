from dataclasses import dataclass, field
from typing import Literal
from functools import cached_property

from quasar_typing.numpy import BoolMatrix, FloatMatrix
from quasar_typing.misc import BackgroundFlux

from .measure_result import MeasureResult

@dataclass(frozen=True)
class MomentParams:
    analyser: object
    from_model: bool = field(
        default=False, 
        kw_only=True,
    )
    bg_flux: BackgroundFlux = field(
        default_factory=lambda: BackgroundFlux({'all', 'em'}),
        kw_only=True
    )
    centre: Literal['wave', 'centroid', 'median', 'mode'] = field(
        default='wave',
        kw_only=True
    )

    @cached_property
    def tot(self) -> MeasureResult:
        return self.analyser.measureStrength.__wrapped__(
            self.analyser,
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )
    
    @cached_property
    def mask(self) -> BoolMatrix:
        return self.analyser.getMask.__wrapped__(
            self.analyser,
            centre=self.centre,
            from_model=self.from_model,
            simplify=False,
        )
    
    @cached_property
    def xs(self) -> FloatMatrix:
        return self.analyser.x[None,:]
    
    @cached_property
    def ys(self) -> FloatMatrix:
        return self.analyser.getY.__wrapped__(
            self.analyser,
            bg_flux=self.bg_flux,
            ensure_not_negative=False,
            simplify=False,
        )
    
    @cached_property
    def dxs(self) -> FloatMatrix:
        return self.analyser.dx[None,:]
    
    @cached_property
    def dys(self) -> FloatMatrix:
        return self.analyser.dy[None,:]
    
    @cached_property
    def m1(self) -> MeasureResult:
        return self.analyser.measureFirstMoment.__wrapped__(
            self.analyser,
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )
    
    @cached_property
    def cm2(self) -> MeasureResult:
        return self.analyser.measureSecondCentralMoment.__wrapped__(
            self.analyser,
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )
    
    @cached_property
    def cm3(self) -> MeasureResult:
        return self.analyser.measureThirdCentralMoment.__wrapped__(
            self.analyser,
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )
    
    @cached_property
    def cm4(self) -> MeasureResult:
        return self.analyser.measureFourthCentralMoment.__wrapped__(
            self.analyser,
            from_model=self.from_model,
            bg_flux=self.bg_flux,
        )



