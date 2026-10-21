__all__ = ["prior_transform"]

from quasar_typing.numpy import FloatVector


def prior_transform(
    u: FloatVector,
    lbs: FloatVector,
    ubs: FloatVector,
) -> FloatVector:
    return lbs + u * (ubs - lbs)