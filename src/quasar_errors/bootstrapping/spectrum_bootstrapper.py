from typing import ClassVar
from quasar_typing.misc import Variant
from .base_bootstrapper import BaseBootstrapper

class SpectrumBootstrapper(BaseBootstrapper):
    variant: ClassVar[Variant] = 'spectrum'
