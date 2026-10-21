__all__ = [
    "BaseBootstrapper",
    "FlexibleBootstrapper",
    "RigidBootstrapper",
    "SpectrumBootstrapper",
    "StandardBootstrapper",
]
from typing import ClassVar, Literal

from .base_bootstrapper import BaseBootstrapper


class FlexibleBootstrapper(BaseBootstrapper):
    variant: ClassVar[Literal["flexible"]] = "flexible"


class RigidBootstrapper(BaseBootstrapper):
    variant: ClassVar[Literal["rigid"]] = "rigid"


class SpectrumBootstrapper(BaseBootstrapper):
    variant: ClassVar[Literal["spectrum"]] = "spectrum"


class StandardBootstrapper(BaseBootstrapper):
    variant: ClassVar[Literal["standard"]] = "standard"