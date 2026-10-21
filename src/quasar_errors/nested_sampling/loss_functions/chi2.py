from numpy import dot, log, pi
from quasar_models.modeling import SequentialModel
from quasar_typing.numpy import CoordsTuple, FloatVector


def evaluate(z: FloatVector) -> float:
    return dot(z, z)


def normalisation(dy: FloatVector) -> float:
    return -0.5 * dy.size * log(2 * pi) - log(dy).sum()


def loglike(
    parameters: FloatVector,
    *,
    model: SequentialModel,
    data: CoordsTuple,
    normalisation: float = 0.0,
) -> float:
    x, y, dy = data
    z = (y - model.evaluate(x, parameters)) / dy
    return normalisation - 0.5 * dot(z, z)