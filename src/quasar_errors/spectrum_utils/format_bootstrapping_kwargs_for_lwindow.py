from logging import getLogger
from typing import Any

from ..bootstrapping import (
    BaseBootstrapper, 
    SpectrumBootstrapper, 
    FlexibleBootstrapper, 
    StandardBootstrapper, 
    RigidBootstrapper,
)
from quasar_models import PowerLawModel, IronModel, BalmerModel

from quasar_utils.continuum_fit_result import ContinuumFitResult

from quasar_typing.numpy import RandomState_
from quasar_typing.astropy import FitterInstance, CompoundModel_
from quasar_typing.logging import Logger_
from quasar_typing.misc import (
    Variant, ModelTypes, BootstrapType, VaryLines, Pool_,
)

logger = getLogger(__name__)

def format_bootstrapping_kwargs_for_lwindow(
    lwindow,
    fitter: FitterInstance,
    *,
    pl: PowerLawModel | None = None,
    fe: IronModel | CompoundModel_[IronModel] | None = None,
    ba: BalmerModel | None = None,
    hg: None = None,

    without_rejections: bool = False,
    without_absorption: bool = False,
    with_neighbours: bool = True,
    variant: Variant = 'standard',
    model_types: ModelTypes | None = None,
    bootstrap_type: BootstrapType = 'model',
    n_sigmas: float = 2.0,
    vary_lines: VaryLines | None = None,
    pool: Pool_ | None = None,
    cfit: ContinuumFitResult | None = None,
    iterations: int = 100,
    random_state: RandomState_ = None,
    renew_rng: bool = False,
    replace_missing: bool = True,
    tqdm_disable: bool = False,
    tqdm_leave: bool = False,
    logger: Logger_ = logger,
) -> tuple[BaseBootstrapper, tuple[Any], dict[str, Any]]:
    """
    Convenience function that formats the arguments for creating a bootstrapper
    instance.

    Parameters
    ----------
    lwindow : quasar_decrypt.LWindow
        The line window to be bootstrapped.
    fitter : astropy.modeling.fitting.Fitter
        The fitter to use for refitting models during bootstrapping.
    pl : PowerLawModel, optional
        The power law model. Retrieved from the parent Spectrum if not given
        and required by the chosen variant.
    fe : IronModel or CompoundModel[IronModel], optional
        The iron model. Retrieved from the parent Spectrum if not given
        and required by the chosen variant.
    ba : BalmerModel, optional
        The Balmer model. Retrieved from the parent Spectrum if not given
        and required by the chosen variant.
    without_rejections : bool
        Whether to exclude pixels sigma-clipped during continuum fitting.
    without_absorption : bool
        Whether to exclude pixels flagged as narrow absorption features.
    with_neighbours : bool
        Whether to include pixels covered by the LWindow's neighbouring
        CWindow instances in addition to those covered by the LWindow itself.
    variant : {'spectrum', 'standard', 'flexible', 'rigid'}
        The bootstrapping variant to use.
    model_types : set[{'pl', 'fe', 'ba', 'hg', 'em'}] | None
        The window types whose models to include. Any type required by the
        chosen variant is added automatically. Defaults to all types if None
        or empty.
    bootstrap_type : {'spectrum', 'model', 'frequentist'}
        The type of bootstrapping to perform, i.e. how to create resampled data.
    n_sigmas : float
        The number of standard deviations used to constrain the power law
        parameter bounds during 'flexible' bootstrapping.
    vary_lines : set['all' | str] | None
        Names of the emission lines to allow to vary during bootstrapping. Only 
        used for 'rigid' bootstrapping. If None, all lines are varied.
    pool : multiprocessing.Pool | None
        The multiprocessing pool to use for parallelization.
    cfit : ContinuumFitResult, optional
        The continuum fit result required for 'flexible' bootstrapping.
        Retrieved from the parent Spectrum's power law windows if not given.
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
    logger : logging.Logger
        The logger this function uses to log messages.
    """
    kwargs = dict(
        pl=pl,
        fe=fe,
        ba=ba,
        hg=hg,
        em=lwindow.getModel(thaw=False),
        usable_pixels=None,
        vary_lines=None,
        sigma_res=lwindow.info.loading['sigma_res'],
        pool=pool,
        info=lwindow.info,
        cfit=cfit,
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
    spectrum = getattr(lwindow, 'spectrum', None)

    ### Models ###

    if not model_types:
        model_types = ModelTypes({'all'})
        msg = f"Empty or missing 'model_types' set! " \
            f"Defaulting to: {model_types=}"
        logger.debug(msg)

    required_model_types: set[str] = set()
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
        kwargs['vary_lines'] = vary_lines or VaryLines({'all'})

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

    for m_type in filter(
        lambda m_type: (kwargs[m_type] is None) and not (m_type == 'em'), 
        model_types,
    ):
        if m_type in required_model_types:
            if spectrum is None:
                msg = f"Could not find parent Spectrum instance for LWindow \
                    needed to retrieve '{m_type}' windows!"
                logger.critical(msg)
                raise ValueError(msg)
            elif (windows := spectrum[m_type]) is None:
                msg = f"Could not find '{m_type}' windows required for \
                    '{variant}' bootstrapping! Cancelling bootstrapping."
                logger.critical(msg)
                raise ValueError(msg)
            elif (model := windows.getModel()) is None:
                msg = f"Could not find '{m_type}' model required for \
                    '{variant}' bootstrapping! Cancelling bootstrapping."
                logger.critical(msg)
                raise ValueError(msg)
            
            kwargs[m_type] = model
        else:
            if spectrum is None:
                continue
            elif (windows := spectrum[m_type]) is None:
                continue
            elif (model := windows.getModel()) is None:
                continue
            kwargs[m_type] = model

    args = (lwindow.x, lwindow.y, lwindow.dy)
    kwargs['usable_pixels'] = lwindow.getMask.__wrapped__(
        lwindow,
        covered=True,
        without_rejections=without_rejections,
        without_absorption=without_absorption,
        with_neighbours=with_neighbours,
        valid=True,
    )[lwindow.mask]
    
    if variant == 'flexible' and kwargs['cfit'] is None:
        if spectrum is None:
            msg = "Missing 'cfit': Could not find parent Spectrum instance \
                needed to retrieve 'pl' windows!"
            logger.critical(msg)
            raise ValueError(msg)
        elif (cont_windows := spectrum['pl']) is None:
            msg = "Missing 'cfit': Could not find 'pl' windows required for \
                'flexible' bootstrapping!"
            logger.critical(msg)
            raise ValueError(msg)
        
        try: 
            cfit = cont_windows.summariseContinuumFitResult()
        except Exception as e:
            msg = f"Missing 'cfit': Failed to get 'cfit' from 'pl' windows: {e}"
            logger.critical(msg)
            raise ValueError(msg)

        kwargs['cfit'] = cfit
    
    return bootstrapper_cls, args, kwargs