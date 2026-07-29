from pydantic.dataclasses import dataclass
from quasar_typing.numpy import FloatMatrix, FloatVector


@dataclass(frozen=True)
class DiscretisedCoords:
    x_bounds: FloatMatrix
    xs: FloatMatrix
    dxs: FloatVector
