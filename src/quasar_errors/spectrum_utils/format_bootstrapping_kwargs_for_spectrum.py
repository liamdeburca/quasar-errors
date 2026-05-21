from logging import getLogger
from typing import Any

from ..bootstrapping import (
    BaseBootstrapper, 
    SpectrumBootstrapper, 
    FlexibleBootstrapper, 
    StandardBootstrapper, 
    RigidBootstrapper,
)

from quasar_typing.numpy import RandomState_
from quasar_typing.astropy import FitterInstance
from quasar_typing.logging import Logger_
from quasar_typing.misc import (
    Variant, ModelTypes, DataTypes, BootstrapType, VaryLines, Pool_,
)

logger = getLogger(__name__)

def format_bootstrapping_kwargs_for_spectrum(
    spectrum,
    fitter: FitterInstance,
    *,
    covered: bool = False,
    without_rejections: bool = False,
    without_absorption: bool = False,
    variant: Variant = 'standard',
    data_types: DataTypes = DataTypes({'all'}),
    model_types: ModelTypes = ModelTypes({'all'}),
    bootstrap_type: BootstrapType = 'model',
    n_sigmas: float = 2.0,
    vary_lines: VaryLines | None = None,
    pool: Pool_ | None = None,
    iterations: int = 100,
    random_state: RandomState_ = None,
    renew_rng: bool = False,
    replace_missing: bool = True,
    tqdm_disable: bool = False,
    tqdm_leave: bool = False,
    logger: Logger_ = logger,
) -> tuple[BaseBootstrapper, tuple[Any, ...], dict[str, Any]]:
    """
    Convenience function that formats the arguments for creating a bootstrapper
    instance.

    Parameters
    ----------
    spectrum : quasar_decrypt.Spectrum
        The spectrum to be bootstrapped.
    covered : bool
        Whether to only include pixels covered by the spectrum's windows.
    without_rejections : bool
        Whether to exclude pixels sigma-clipped during continuum fitting.
    without_absorption : bool
        Whether to exclude pixels flagged as narrow absorption features.
    variant : {'spectrum', 'standard', 'flexible', 'rigid'}
        The bootstrapping variant to use. 
    data_types : set[{'pl', 'fe', 'ba', 'hg', 'em'}] | None
        The windows whose data to include in the bootstrapping data.
    model_types : set[{'pl', 'fe', 'ba', 'hg', 'em'}] | None
        The windows whose models to include in the bootstrapping data.
    bootstrap_type : {'spectrum', 'model', 'frequentist'}
        The type of bootstrapping to perform, i.e. how to create resampled data.
    n_sigmas : float
        The number of standard deviations used to constrain the power law
        parameter bounds during 'flexible' bootstrapping.
    vary_lines : set['all' | float] | None
        The emission lines to allow to vary during bootstrapping. Only used for
        'rigid' bootstrapping. If None, all lines are varied.
    pool : multiprocessing.Pool | None
        The multiprocessing pool to use for parallelization.
    iterations : int
        The number of bootstrap iterations to perform.
    random_state : numpy.random.Generator | numpy.random.RandomState | int | None
        The random state to use for reproducibility.
    renew_rng : bool
        Whether to renew the random number generator at the start of the 
        bootstrapping process.
    replace_missing : bool
        Whether to replace missing/rejected pixels.
    tqdm_disable : bool
        Whether to disable tqdm progress bars.
    tqdm_leave : bool
        Whether to leave tqdm progress bars after completion.
    fitter : astropy.modeling.fitting.Fitter
        The fitter to use for refitting models during bootstrapping.
    logger : logging.Logger
        The logger this functions uses to log messages.
    """
    kwargs = dict(
        pl=None,
        fe=None,
        ba=None,
        em=None,
        usable_pixels=None,
        vary_lines=None,
        sigma_res=spectrum.info.loading['sigma_res'],
        pool=pool,
        info=spectrum.info,
        cfit=None,
        fitter=fitter,
        iterations=iterations,
        random_state=random_state,
        renew_rng=renew_rng,
        bootstrap_type=bootstrap_type,
        n_sigmas=n_sigmas,
        replace_missing=replace_missing,
        tqdm_disable=tqdm_disable,
        tqdm_leave=tqdm_leave,
    )

    required_model_types = set()
    if variant == 'spectrum':
        # No refitting
        bootstrapper_cls = SpectrumBootstrapper
    elif variant == 'standard':
        # All models are refit
        bootstrapper_cls = StandardBootstrapper
    elif variant == 'flexible':
        # Power law and emission lines are refit
        bootstrapper_cls = FlexibleBootstrapper
        required_model_types.add('pl')
        required_model_types.add('em')
    else:
        # Emission lines are refit
        bootstrapper_cls = RigidBootstrapper
        required_model_types.add('em')
        kwargs['vary_lines'] = vary_lines or {'all'}

    for r_type in required_model_types:
        if r_type in model_types:
            continue

        logger.warning(
            "Required model type '%s' not included in 'model_types' despite "
            "using '%s' bootstrapping! Adding it.",
            r_type,
            variant,
        )
        model_types.add(r_type)

        if spectrum[r_type] is None:
            msg = f"Could not find '{r_type}' windows required for \
                '{variant}' bootstrapping! Cancelling bootstrapping."
            logger.critical(msg)
            raise ValueError(msg)
        elif spectrum[r_type].getModel() is None:
            msg = f"Could not find '{r_type}' model required for \
                '{variant}' bootstrapping! Cancelling bootstrapping."
            logger.critical(msg)
            raise ValueError(msg)
        
    for m_type in model_types:
        if (windows := spectrum[m_type]) is None:
            continue
        if (model := windows.getModel()) is None:
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
    if variant == 'flexible':
        kwargs['cfit'] = spectrum.summariseContinuumFitResult()
    
    return bootstrapper_cls, args, kwargs