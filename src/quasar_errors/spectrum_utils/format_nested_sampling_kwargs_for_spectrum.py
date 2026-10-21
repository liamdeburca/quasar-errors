__all__ = ["format_nested_sampling_kwargs_for_spectrum"]

from logging import getLogger
from typing import Any, Literal, Protocol

from quasar_typing.literals import Variant
from quasar_typing.logging import LoggerType
from quasar_typing.multiprocessing import PoolType
from quasar_typing.numpy import FloatVector, RandomStateType
from quasar_typing.string_selections import (
    DataTypes,
    ModelTypes,
)
from quasar_utils.setup import Info

from quasar_errors.nested_sampling import (
    BaseNestedSampler,
    UniformNestedSampler,
)

logger = getLogger(__name__)


class SpectrumProtocol(Protocol):
    _x: FloatVector
    _y: FloatVector
    _dy: FloatVector
    info: Info


def format_nested_sampling_kwargs_for_spectrum(
    spectrum: SpectrumProtocol,
    *,
    covered: bool = False,
    without_rejections: bool = False,
    without_absorption: bool = False,
    variant: Variant = "standard",
    data_types: DataTypes | None = None,
    model_types: ModelTypes | None = None,
    nested_type: Literal["uniform"] = "uniform",
    pool: PoolType | None = None,
    iterations: int = 100,
    random_state: RandomStateType = None,
    replace_missing: bool = True,
    tqdm_disable: bool = False,
    tqdm_leave: bool = False,
    logger: LoggerType = logger,
) -> tuple[BaseNestedSampler, tuple[Any, ...], dict[str, Any]]:
    if data_types is None:
        data_types = DataTypes({"all"})
    if model_types is None:
        model_types = ModelTypes({"all"})

    kwargs = {
        "pl": None,
        "fe": None,
        "ba": None,
        "em": None,
        "hg": None,
        "usable_pixels": None,
        "info": spectrum.info,
        "pool": pool,
        "iterations": iterations,
        "random_state": random_state,
        "replace_missing": replace_missing,
        "tqdm_disable": tqdm_disable,
        "tqdm_leave": tqdm_leave,
    }

    match nested_type:
        case "uniform":
            nested_sampler_cls = UniformNestedSampler

    # ! Something is wrong here... Looping over an empty set???
    required_model_types = set()
    for r_type in required_model_types:
        if model_types(r_type):
            continue

        logger.warning(
            "Required model type '%s' not included in 'model_types' despite "
            "using '%s' bootstrapping! Adding it.",
            r_type,
            variant,
        )
        model_types.add(r_type)

        if spectrum[r_type] is None:
            msg = f"Could not find '{r_type}' windows required for '{variant}' nested sampling! Cancelling nested sampling."
            logger.critical(msg)
            raise ValueError(msg)
        elif spectrum[r_type].getModel() is None:
            msg = f"Could not find '{r_type}' model required for '{variant}' nested sampling! Cancelling nested sampling."
            logger.critical(msg)
            raise ValueError(msg)

    for m_type in model_types:
        if (windows := spectrum[m_type]) is not None and (
            model := windows.getModel()
        ) is not None:
            kwargs[m_type] = model

    ### Data types // Windows ###
    for d_type in filter(
        lambda d_type: d_type not in model_types,
        data_types,
    ):
        msg = f"Required model type '{d_type}' is not included in 'data_types'! This may lead to inaccurate error estimation."
        logger.warning(msg)

    args = (spectrum._x, spectrum._y, spectrum._dy)
    kwargs["usable_pixels"] = spectrum.getMask.__wrapped__(
        spectrum,
        data_types=data_types,
        covered=covered,
        without_rejections=without_rejections,
        without_absorption=without_absorption,
        valid=True,
    )

    return nested_sampler_cls, args, kwargs
