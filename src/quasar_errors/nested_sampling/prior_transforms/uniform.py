from quasar_typing.numpy import FloatVector
from ..pseudo_compound_model import PseudoCompoundModel

def prior_transform(
    u: FloatVector,
    model: PseudoCompoundModel,
) -> FloatVector:
    assert u.size == model.n_free

    p = model.parameters.copy()
    p[model.free_mask] = model.free_lbs + u * (model.free_ubs - model.free_lbs)

    if model.n_tied > 0:
        model.setParameters(p)
        for idx, func in model.tied_functions.items():
            p[idx] = func(model)
        model.setParameters(p)

    return p