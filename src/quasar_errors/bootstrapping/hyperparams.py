from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.numpy import RandomState_
from quasar_typing.astropy import FitterInstance
from quasar_typing.misc import BootstrapType, VaryLines

@dataclass(frozen=True)
class HyperParams:
    fitter: FitterInstance
    iterations: int
    random_state: RandomState_
    renew_rng: bool
    bootstrap_type: BootstrapType
    replace_missing: bool
    vary_lines: VaryLines
    tqdm_disable: bool

    random_seed: dict = field(init=False)

    def __post_init__(self):
        self.random_seed = self.random_state.get_state()