from pydantic.dataclasses import dataclass
from quasar_typing.literals import BootstrapType
from quasar_typing.numpy import RandomStateType
from quasar_typing.string_selections import VaryLines
from quasar_utils.setup import FitterKwargs


@dataclass(frozen=True)
class HyperParams:
    fitter_kwargs: FitterKwargs
    iterations: int
    random_state: RandomStateType
    bootstrap_type: BootstrapType
    replace_missing: bool
    vary_lines: VaryLines
    tqdm_disable: bool