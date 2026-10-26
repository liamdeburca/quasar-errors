from collections.abc import Iterator
from typing import Optional

from numpy import isfinite, median
from numpy.random import RandomState
from quasar_models.modeling import PrepareModel
from quasar_typing.astropy import ModelType
from quasar_typing.numpy import (
    BoolVector,
    FittableFloatVector,
    FloatMatrix,
    FloatVector,
)
from quasar_utils.resolving import mask_kernels


def _replace_missing_helper(
    x: FittableFloatVector,
    y: FloatVector,
    dy: FloatVector,
    full_model: Optional[ModelType],
    kernels: FloatMatrix | None,
    is_logbinned: bool | None,
    bootstrap_type: str,   
) -> tuple[BoolVector, list[str]]:
    msgs: list[str] = []

    y_valid_mask = isfinite(y)
    dy_valid_mask = isfinite(dy) & (dy > 0)
    is_valid = y_valid_mask & dy_valid_mask

    n_pix: int = x.size
    n_y_valid: int = y_valid_mask.sum()
    n_dy_valid: int = dy_valid_mask.sum()
    n_valid: int = is_valid.sum()
    msgs.append(
        f"{n_valid}/{n_pix} (thereof: {n_y_valid} ('y'), {n_dy_valid} ('dy'))"
    )

    if is_valid.all():
        msgs.append("All pixels are valid -> returning early!")
        return is_valid, msgs
    elif full_model is None:
        msgs.append("No full model provided -> cannot replace missing values.")
        return is_valid, msgs
    elif bootstrap_type == "frequentist":
        msgs.append("Performing frequentist bootstrap -> skipping replacement")
        return is_valid, msgs


    if y_valid_mask.all():
        msgs.append("All `y` values are valid!")
    else:
        msgs.append("Replacing missing `y` values with model predictions.")
        preparer = PrepareModel(
            x=x[y_valid_mask],
            model=full_model,
            kernels=mask_kernels(kernels, y_valid_mask),
            is_logbinned=is_logbinned,
        )
        with preparer as (seq_model, _):
            y[~y_valid_mask] = seq_model(x)


    if dy_valid_mask.all():
        msgs.append("All `dy` values are valid!")
    elif not dy_valid_mask.any():
        msgs.append("No valid `dy` values available! Replacing with: 1.0.")
        dy[:] = 1.0
    else:
        dy[~dy_valid_mask] = med = median(dy[dy_valid_mask])
        msgs.append(
            f"Replacing {(~dy_valid_mask).sum()} missing `dy` values. "
            f"Replacing with median: {med}"
        )

    is_valid = isfinite(y) & isfinite(dy) & (dy > 0)
    assert is_valid.all()

    return is_valid, msgs


def replace_missing_helper(
    *,
    x: FittableFloatVector,
    y: FloatVector,
    dy: FloatVector,
    eval_mask: BoolVector,
    comp_mask: BoolVector,
    full_model: Optional[ModelType],
    kernels: FloatMatrix | None,
    is_logbinned: bool | None,
    bootstrap_type: str,
) -> list[str]:
    """
    Helper function for replacing missing values within the `eval_mask` mask.

    Modifications on the `y` and `dy` float arrays, and the `comp_mask` boolean
    array, are performed in-place.

    Parameters
    ----------
    x : FloatVector
        Array of rest wavelength values.
    y : FloatVector
        Array of flux density values.
    dy : FloatVector
        Array of flux density uncertainties.
    eval_mask : BoolVector
        Boolean mask indicating which elements are used to evaluate the model.
    comp_mask : BoolVector
        Boolean mask indicating which elements are used for comparing the model
        to data.
    full_model : ModelType | None
        The model.
    kernels : FloatMatrix | None
        Matrix of kernels used to correct the intrinsic model for instrumental 
        resolution.
    is_logbinned : bool | None
        Whether the spectrum is log-binned. If None, it is inferred from the 
        `x` array.
    bootstrap_type : str
        The type of bootstrap being performed.

    Returns
    -------
    list[str]
        A list of messages describing the replacement process.
    """
    mask = eval_mask & comp_mask
    is_valid, msgs = _replace_missing_helper(
        x[mask], y[mask], dy[mask],
        full_model=full_model,
        kernels=mask_kernels(kernels, mask),
        is_logbinned=is_logbinned,
        bootstrap_type=bootstrap_type,
    )
    msgs.append("Replacement complete: updating `comp` mask!")
    comp_mask[mask] = is_valid

    return msgs


def generate_samples_helper(
    *,
    x: FloatVector,
    y: FloatVector,
    dy: FloatVector,
    eval_mask: BoolVector,
    comp_mask: BoolVector,
    full_model: Optional[ModelType],
    bootstrap_type: str,
    kernels: FloatMatrix,
    is_logbinned: bool | None,
    iterations: int,
    rng: RandomState,
) -> tuple[list[str], Iterator[FloatVector]]:
    mask = eval_mask & comp_mask
    n_points: int = mask.sum()

    msgs: list[str] = []
    if bootstrap_type == "frequentist" or bootstrap_type == "model":
        if full_model is None:
            raise ValueError(
                f"Bootstrap type '{bootstrap_type}' requires a model."
            )

        msgs.append(f"Using '{bootstrap_type}' bootstrap.")

        preparer = PrepareModel(
            x=x,
            model=full_model,
            eval_mask=eval_mask,
            comp_mask=comp_mask,
            kernels=kernels,
            is_logbinned=is_logbinned,
        )
        f = y.copy("C")
        with preparer as (seq_model, _):
            f[eval_mask] = seq_model(x[eval_mask])

        if bootstrap_type == "frequentist":
            # Only consider differences for the points used to evaluate the model
            diffs = y[mask] - f[mask]
            def func() -> FloatVector:
                """
                Shuffle data-model differences on the `eval_mask` mask.
                """
                out = f.copy("C")
                out[mask] += rng.choice(diffs, size=n_points, replace=True)
                return out
        else:
            def func() -> FloatVector:
                """
                Add Gaussian noise to the model predictions on the `eval_mask` mask.
                """
                out = f.copy("C")
                out[mask] += rng.normal(scale=dy[mask], size=n_points)
                return out
    else:
        msgs.append(f"Defaulted to 'spectrum' bootstrap from: {bootstrap_type=}")
        def func() -> FloatVector:
            """
            Resample the spectrum with replacement on the `eval_mask` mask.
            """
            out = y.copy("C")
            out[mask] += rng.normal(scale=dy[mask], size=n_points)
            return out

    return msgs, (func() for _ in range(iterations))