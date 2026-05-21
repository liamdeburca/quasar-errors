from typing import ClassVar
from quasar_typing.misc import Variant
from .base_bootstrapper import BaseBootstrapper

class FlexibleBootstrapper(BaseBootstrapper):
    variant: ClassVar[Variant] = 'flexible'