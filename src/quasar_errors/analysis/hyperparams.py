from pydantic.dataclasses import dataclass
from quasar_typing.misc import FWHMStrategy


@dataclass(frozen=True)
class HyperParams:
    res: int
    render_width: float
    fwhm_strategy: FWHMStrategy
    exact: bool
    v_int: float
    ipv_int: float
    dx_int: float
