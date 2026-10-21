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
    msgs: list[str] = []

    mask = eval_mask & comp_mask
    _mask = comp_mask[eval_mask]
    n_pix: int = _mask.size # no. of pixels used for evaluation
    n_valid: int = _mask.sum() # no. of valid pixels used for evaluation
    msgs.append(f"{n_pix=}, thereof {n_valid=}")

    if _mask.all():
        # All pixels are valid
        msgs.append("All pixels are valid -> returning early!")
        return msgs

    if bootstrap_type == "frequentist":
        msgs.append("Performing frequentist bootstrap -> skipping replacement")
        return msgs

    ###

    valid_y = mask & isfinite(y)
    invalid_y = mask & ~isfinite(y)
    if valid_y[comp_mask].all():
        msgs.append("No missing `y` values!")
    elif full_model is None:
        msgs.append("No full model provided -> cannot replace missing `y` values.")
    else:
        msgs.append(f"Replacing {invalid_y[mask].sum()} missing `y` values.")
        preparer = PrepareModel(
            x=x,
            model=full_model,
            eval_mask=eval_mask,
            comp_mask=comp_mask,
            kernels=kernels,
            is_logbinned=is_logbinned,
        )
        with preparer as (seq_model, _):
            f = seq_model(x[eval_mask]) # Only evaluate on `eval_mask`
            m = invalid_y[eval_mask]
            y[eval_mask][m] = f[m]

    ###

    valid_dy = mask & (isfinite(dy) & (dy > 0))
    invalid_dy = mask & ~(isfinite(dy) & (dy > 0))
    if valid_dy[comp_mask].all():
        msgs.append("No missing `dy` values!")
    else:
        msgs.append(f"Replacing {invalid_dy[mask].sum()} missing `dy` values(s).")
        med = median(dy[eval_mask][valid_dy[eval_mask]])
        msgs.append(f"Median of valid `dy` values: {med}")
        dy[eval_mask][invalid_dy[eval_mask]] = med

    msgs.append("Replacement complete: updating `comp` mask!")
    comp_mask[eval_mask] = True

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