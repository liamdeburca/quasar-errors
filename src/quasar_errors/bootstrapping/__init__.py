__all__ = [
    "BaseBootstrapper",
    "FlexibleBootstrapper",
    "RigidBootstrapper",
    "SpectrumBootstrapper",
    "StandardBootstrapper",
]
from typing import ClassVar

from quasar_typing.misc import Variant

from .base_bootstrapper import BaseBootstrapper


class FlexibleBootstrapper(BaseBootstrapper):
    variant: ClassVar[Variant] = "flexible"


class RigidBootstrapper(BaseBootstrapper):
    variant: ClassVar[Variant] = "rigid"


class SpectrumBootstrapper(BaseBootstrapper):
    variant: ClassVar[Variant] = "spectrum"


class StandardBootstrapper(BaseBootstrapper):
    variant: ClassVar[Variant] = "standard"
