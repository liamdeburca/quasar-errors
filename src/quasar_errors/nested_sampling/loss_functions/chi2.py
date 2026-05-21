from numpy import dot, log, pi
from quasar_typing.numpy import FloatVector, CoordsTuple
from ..pseudo_compound_model import PseudoCompoundModel

def evaluate(z: FloatVector) -> float:
    return dot(z, z)

def normalisation(dy: FloatVector) -> float:
    return -0.5 * dy.size * log(2 * pi) - log(dy).sum()

def loglike(
    parameters: FloatVector,
    *,
    model: PseudoCompoundModel,
    data: CoordsTuple,
    normalisation: float = 0.0,
) -> float:
    p = model.upsample(parameters)
    x, y, dy = data
    z = (y - model.forward(x, p)) / dy
    return normalisation + dot(z, z)