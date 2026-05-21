__all__ = ['format_nested_sampling_kwargs_for_spectrum']

from logging import getLogger
from typing import Any, Protocol, Literal

from quasar_typing.numpy import RandomState_, FloatVector
from quasar_typing.logging import Logger_
from quasar_typing.misc import (
    Variant, ModelTypes, DataTypes, Pool_,
)

from quasar_utils.setup import Info

from ..nested_sampling import BaseNestedSampler, UniformNestedSampler

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
    variant: Variant = 'standard',
    data_types: DataTypes = DataTypes({'all'}),
    model_types: ModelTypes = ModelTypes({'all'}),
    nested_type: Literal['uniform'] = 'uniform',
    pool: Pool_ | None = None,
    iterations: int = 100,
    random_state: RandomState_ = None,
    renew_rng: bool = False,
    replace_missing: bool = True,
    tqdm_disable: bool = False,
    tqdm_leave: bool = False,
    logger: Logger_ = logger,
) -> tuple[BaseNestedSampler, tuple[Any, ...], dict[str, Any]]:
    kwargs = dict(
        pl=None,
        fe=None,
        ba=None,
        em=None,
        hg=None,
        usable_pixels=None,
        info=spectrum.info,
        pool=pool,
        iterations=iterations,
        random_state=random_state,
        renew_rng=renew_rng,
        replace_missing=replace_missing,
        tqdm_disable=tqdm_disable,
        tqdm_leave=tqdm_leave,
    )

    match nested_type:
        case 'uniform':
            nested_sampler_cls = UniformNestedSampler
        

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
            msg = "Could not find '%s' windows required for '%s' nested "\
                "sampling! Cancelling nested sampling.".format(
                    r_type, variant,
                )
            logger.critical(msg)
            raise ValueError(msg)
        elif spectrum[r_type].getModel() is None:
            msg = "Could not find '%s' model required for '%s' nested "\
                "sampling! Cancelling nested sampling.".format(
                    r_type, variant,
                )
            logger.critical(msg)
            raise ValueError(msg)
        
    for m_type in model_types:
        if (windows := spectrum[m_type]) is None:
            continue
        elif (model := windows.getModel()) is None:
            continue
        kwargs[m_type] = model

    ### Data types // Windows ###
    for d_type in data_types:
        if d_type in model_types:
            continue

        logger.warning(
            f"Required model type '{d_type}' is not included in "
            "'data_types'! This may lead to inaccurate error estimation. "
        )

    args = (spectrum._x, spectrum._y, spectrum._dy)
    kwargs['usable_pixels'] = spectrum.getMask.__wrapped__(
        spectrum,
        data_types=data_types,
        covered=covered,
        without_rejections=without_rejections,
        without_absorption=without_absorption,
        valid=True,
    )
    
    return nested_sampler_cls, args, kwargs