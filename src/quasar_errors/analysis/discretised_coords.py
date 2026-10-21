from typing import TypedDict

from quasar_typing.numpy import FloatMatrix, FloatVector


class DiscretisedCoords(TypedDict):
    x_bounds: FloatMatrix
    xs: FloatMatrix
    dxs: FloatVector
