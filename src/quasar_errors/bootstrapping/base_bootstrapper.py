import logging
from typing import Self, Iterable, Iterator, Callable, ClassVar
from dataclasses import field
from numpy import isfinite, zeros_like, float64, argwhere, invert, median, ascontiguousarray
from tqdm import tqdm

from pydantic.dataclasses import dataclass

from quasar_utils.setup import Info
from quasar_utils.decorators import validate_call, apply_info_to_method

from quasar_models import PowerLawModel, IronModel, BalmerModel, GaussianModel
from quasar_models.line import _VProfileCopy

from quasar_typing.astropy import FitterInstance, Model_, CompoundModel_
from quasar_typing.numpy import BoolVector, FloatVector, RandomState_, CoordsTuple
from quasar_typing.misc import BootstrapType, Variant, VaryLines, Pool_

from quasar_utils.continuum_fit_result import ContinuumFitResult

from ..model_samples import Sample
from ..error_result import ErrorResult, ErrorResultBuilder

type MappedFunc = Callable[[FloatVector], Sample]

logger = logging.getLogger(__name__)

@dataclass
class BaseBootstrapper:
    """
    Core bootstrapping class designed for class inheritance.

    To create a new bootstrapper class, simply inherit from this class, decorate
    the new class with @dataclass, and set the 'variant' class variable. 
    """
    variant: ClassVar[Variant]

    x: FloatVector
    y: FloatVector
    dy: FloatVector
    
    usable_pixels: BoolVector | None = None
    sigma_res: float | None = None
    info: Info = None

    cfit: ContinuumFitResult | None = None
    pl: PowerLawModel | None = None
    fe: IronModel | CompoundModel_[IronModel] | None = None
    ba: BalmerModel | None = None
    hg: Model_ | None = None
    em: GaussianModel | _VProfileCopy | CompoundModel_[GaussianModel | _VProfileCopy] | None = None

    pool: Pool_ | None = None

    # Hyperparameters
    fitter: FitterInstance | None = None
    iterations: int | None = None
    random_state: RandomState_ | None = None
    renew_rng: bool | None = None
    bootstrap_type: BootstrapType | None = None
    n_sigmas: float | None = None
    replace_missing: bool | None = None
    vary_lines: VaryLines | None = None
    tqdm_disable: bool | None = None
    tqdm_leave: bool | None = None

    # Set by __post_init__
    random_seed: dict = field(init=False)
    y_bg: FloatVector = field(init=False)
    full_model: Model_ | None = field(init=False)
    fitted_model: Model_ | None = field(init=False)
    base_sample: Sample = field(init=False)
    mapped_func: MappedFunc = field(init=False)
    mapper: Callable[[MappedFunc, Iterable[FloatVector]], Iterator[Sample]] \
        = field(init=False)

    def __post_init__(self):
        # Resolve usable_pixels from data if not provided.
        if self.usable_pixels is None:
            self.usable_pixels = (
                isfinite(self.x) & isfinite(self.y) & isfinite(self.dy)
                & (self.dy > 0)
            )

        if self.sigma_res is None:
            self.sigma_res = self.info.loading['sigma_res']

        self.setHyperParams(
            fitter=self.fitter,
            iterations=self.iterations,
            random_state=self.random_state,
            renew_rng=self.renew_rng,
            bootstrap_type=self.bootstrap_type,
            replace_missing=self.replace_missing,
            vary_lines=self.vary_lines,
            tqdm_disable=self.tqdm_disable,
            tqdm_leave=self.tqdm_leave,
        )

        self.y_bg = zeros_like(self.y, dtype=float64)
        self.full_model = self._compute_full_model()

        if self.replace_missing and not self.usable_pixels.all():
            self.replaceMissing()

        self.fitted_model = self._compute_fitted_model()
        self.base_sample = self._compute_base_sample()
        self.mapped_func = self._compute_mapped_func()
        self.mapper = map if self.pool is None else self.pool.map

        logger.debug(
            "%d/%d usable pixels, bootstrap_type=%r, iterations=%d.",
            self.usable_pixels.sum(),
            len(self.usable_pixels),
            self.bootstrap_type,
            self.iterations,
        )

    def _compute_base_sample(self) -> Sample:
        """
        Corresponding 'Sample' object for the original data.
        """
        x, y, dy = self.coords

        if self.full_model is None:
            logger.debug(
                "creating base sample w/o model; variant=%r, npix=%d.",
                self.variant, x.size,
            )
            return Sample.fromSpectrumSample(x, y, dy)

        logger.debug(
            "creating base sample w/ model; variant=%r, npix=%d.",
            self.variant, x.size,
        )
        return Sample.fromModelSample(x, y, dy, model=self.full_model)

    def _compute_full_model(self) -> Model_ | None:
        """
        Full model, i.e. the sum of all available model components.
        If none are available, returns None.
        """
        submodels = []
        for component in (self.pl, self.fe, self.ba, self.hg, self.em):
            if component is None:
                continue
            elif isinstance(component, Iterable):
                submodels.extend(component)
            else:
                submodels.append(component)

        if not submodels:
            logger.debug("no model components provided, returning None.")
            return None

        logger.debug("built from %d submodel(s).", len(submodels))
        return sum(submodels[1:], start=submodels[0])

    def _compute_fitted_model(self) -> Model_ | None:
        """
        The model to be refitted at each iteration.
        """
        def filter_func(m: Model_ | None) -> bool: return m is not None

        match self.variant:
            case 'spectrum':
                logger.debug("variant=%r, no refitting.", self.variant)
                bgs = ()
                fitted_model = None
            case 'standard':
                if self.full_model is None:
                    msg = "Bootstrapping variant 'standard' requires a fitted \
                        model."
                    logger.critical(msg)
                    raise ValueError(msg)

                logger.debug("variant=%r, refitting full model.", self.variant)
                bgs = ()
                fitted_model = self.full_model
            case 'flexible':
                if self.pl is None or self.em is None:
                    msg = "Bootstrapping variant 'flexible' requires both a \
                        power law and an emission line model. Missing: {}" \
                        .format({'pl': self.pl is None, 'em': self.em is None})
                    logger.critical(msg)
                    raise ValueError(msg)
                if self.cfit is None:
                    msg = "Bootstrapping variant 'flexible' requires a \
                        continuum fit result."
                    logger.critical(msg)
                    raise ValueError(msg)
                
                pl = self.pl.copy()

                s = self.cfit.std[0]
                pl.flux.bounds = (
                    max(pl.flux.bounds[0], pl.flux.value - self.n_sigmas * s),
                    min(pl.flux.bounds[1], pl.flux.value + self.n_sigmas * s),
                )
                s = self.cfit.std[1]
                pl.alpha.bounds = (
                    max(pl.alpha.bounds[0], pl.alpha.value - self.n_sigmas * s),
                    min(pl.alpha.bounds[1], pl.alpha.value + self.n_sigmas * s),
                )

                bgs = filter(filter_func, (self.fe, self.ba, self.hg))
                fitted_model = pl + self.em
                logger.debug(
                    "variant=%r, refitting pl + em; background: fe=%r, ba=%r, "
                    "hg=%r.",
                    self.variant, 
                    self.fe is not None, 
                    self.ba is not None, 
                    self.hg is not None,
                )
            case 'rigid':
                if self.em is None:
                    msg = "Bootstrapping variant 'rigid' requires an emission \
                        line model."
                    logger.critical(msg)
                    raise ValueError(msg)

                bgs = filter(filter_func, (self.pl, self.fe, self.ba, self.hg))
                ms = (self.em,) if self.em.n_submodels == 1 else self.em
                
                submodels = []
                extra_bgs = []
                for m in ms:
                    if self.vary_lines(m.pure_name):
                        submodels.append(m)
                    else:
                        extra_bgs.append(m)

                if not submodels:
                    fitted_model = self.em
                    logger.debug(
                        "variant=%r, all emission lines frozen; "
                        "refitting full emission line model.",
                        self.variant,
                    )
                else:
                    fitted_model = sum(submodels[1:], start=submodels[0])
                    bgs = (*bgs, *extra_bgs)
                    logger.debug(
                        "variant=%r, refitting %d/%d emission "
                        "line(s); vary_lines=%r.",
                        self.variant, len(submodels),
                        len(submodels) + len(extra_bgs),
                        self.vary_lines,
                    )

        x = self.x[self.usable_pixels]
        self.y_bg[:] = 0.0
        self.y_bg[self.usable_pixels] = sum((f(x) for f in bgs), start=0.0)

        return fitted_model

    def _compute_mapped_func(self) -> MappedFunc:
        """
        Returns the function to be mapped over the generated samples, i.e. the
        function that creates a 'Sample' object from a resampled flux density
        array.
        """
        x = ascontiguousarray(self.x[self.usable_pixels])
        dy = ascontiguousarray(self.dy[self.usable_pixels])
        y_bg = ascontiguousarray(self.y_bg[self.usable_pixels])

        if self.variant == 'spectrum':
            logger.debug(
                "variant=%r, npix=%d, building function w/o fitting",
                self.variant, x.size,
            )
            def mapped_func_inner(y: FloatVector) -> Sample:
                nonlocal x, dy, y_bg
                return Sample.fromSpectrumSample(
                    x, y - y_bg, dy, weight=1.0,
                )

        else:
            logger.debug(
                "variant=%r, npix=%d, building function w/ fitting",
                self.variant, x.size,
            )
            model = self.fitted_model.copy()
            fitter = self.fitter
            def mapped_func_inner(y: FloatVector) -> Sample:
                nonlocal x, dy, y_bg, model, fitter
                _y = y - y_bg
                _ = fitter(model, x, _y, dy, inplace=True)
                return Sample.fromModelSample(
                    x, _y, dy, model, weight=1.0,
                )

        return mapped_func_inner
    
    @property
    def coords(self) -> CoordsTuple:
        return (
            ascontiguousarray(self.x[self.usable_pixels]),
            ascontiguousarray(self.y[self.usable_pixels]),
            ascontiguousarray(self.dy[self.usable_pixels]),
        )

    @apply_info_to_method(subjects=('nonlinear', 'error'))
    def setHyperParams(
        self,
        *,
        fitter: FitterInstance | None = None,
        iterations: int | None = None,
        random_state: RandomState_ | None = None,
        renew_rng: bool | None = None,
        bootstrap_type: BootstrapType | None = None,
        replace_missing: bool | None = None,
        vary_lines: VaryLines | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
    ) -> None:
        """
        ** INFO APPLIED TO METHOD **

        Resolves and assigns hyperparameter fields from the provided values,
        falling back to defaults drawn from 'self.info' via
        @apply_info_to_method for any argument left as None:
        """
        self.fitter = fitter
        self.iterations = iterations
        self.random_state = random_state
        self.renew_rng = renew_rng
        self.bootstrap_type = bootstrap_type
        self.replace_missing = replace_missing
        self.vary_lines = vary_lines
        self.tqdm_disable = tqdm_disable
        self.tqdm_leave = tqdm_leave
        self.random_seed = self.random_state.get_state()
        logger.debug(
            "fitter=%r, iterations=%d, random_state=%r, renew_rng=%r, "
            "bootstrap_type=%r, replace_missing=%r, vary_lines=%r, "
            "tqdm_disable=%r, tqdm_leave=%r.",
            self.fitter.__class__.__name__,
            self.iterations,
            self.random_state,
            self.renew_rng,
            self.bootstrap_type,
            self.replace_missing,
            self.vary_lines,
            self.tqdm_disable,
            self.tqdm_leave,
        )

    def replaceMissing(self) -> Self:
        """
        Replaces missing pixels, i.e. non-finite rest wavelength values, non-
        finite or unusable flux density values, and non-positive flux density
        uncertainty values.

        Changes are made according to the following rules:

        1.  Missing rest wavelength values are replaced using the input velocity
            resolution, 'self.sigma_res', according to the equation:

                x[j] = x[i] * (1 + self.sigma_res)**(i-j)

        2.  Missing flux density values are replaced using the fitted model(s).

        3.  Missing flux density uncertainty values are replaced using the
            median of the valid values.

        Notes
        -----
        If 'self.bootstrap_type' is 'frequentist', i.e. the data-model
        differences are shuffled at each iteration, this method does nothing.

        In the event that 'self.bootstrap_type' is 'spectrum', the pixels
        lacking both valid flux density and flux density uncertainty values,
        will have their replaced uncertainties amplified by a factor of
        sqrt(2).
        """
        logger.debug(
            "%d/%d non-usable pixel(s).",
            invert(self.usable_pixels).sum(), len(self.usable_pixels),
        )

        if self.usable_pixels.all():
            return self

        if self.bootstrap_type == 'frequentist':
            logger.debug(
                "bootstrap_type=%r, skipping replacement.",
                self.bootstrap_type,
            )
            return self

        if self.full_model is None:
            logger.warning(
                "no model available; missing flux density values will not be "
                "replaced."
            )
            return self

        msg = ""

        valid_x = isfinite(self.x)
        invalid_x = invert(valid_x)
        if valid_x.all():
            msg += "no missing 'x' values, "
        else:
            msg += f"replacing {invalid_x.sum()} missing 'x' value(s), "
            valid_indices = argwhere(valid_x).flatten()
            invalid_indices = argwhere(invalid_x).flatten()

            first = valid_indices[0]
            powers = invalid_indices - first

            self.x[invalid_x] = self.x[first] * (1 + self.sigma_res)**powers

        valid_y = isfinite(self.y) & self.usable_pixels
        invalid_y = invert(valid_y)
        if valid_y.all():
            msg += "no missing 'y' values, "
        else:
            msg += f"replacing {invalid_y.sum()} missing 'y' value(s), "
            self.y[invalid_y] = self.full_model(self.x[invalid_y])

        valid_dy = isfinite(self.dy) & (self.dy > 0)
        invalid_dy = invert(valid_dy)
        if valid_dy.all():
            msg += "no missing 'dy' values."
        else:
            msg += f"replacing {invalid_dy.sum()} missing 'dy' value(s), "
            self.dy[invalid_dy] = median(self.dy[valid_dy])

        if self.bootstrap_type == 'spectrum' and invalid_y.any():
            # When both the flux density and flux density uncertainties
            # have been replaced.
            self.dy[invalid_y] *= 2**0.5

        # Make every element True.
        self.usable_pixels[:] = True
        msg += f"complete! All {self.usable_pixels.size} pixel(s) are usable"
        logger.debug(msg)

        return self

    def generateSamples(self) -> Iterator[FloatVector]:
        """
        Creates an iterator that generates resampled flux density arrays
        according to 'self.bootstrap_type' and the available model components.
        """
        logger.debug(
            "bootstrap_type=%r, iterations=%d, renew_rng=%r.",
            self.bootstrap_type,
            self.iterations,
            self.renew_rng,
        )
        if self.renew_rng:
            self.random_state.set_state(self.random_seed)

        x, y, dy = self.coords

        n_points: int = self.usable_pixels.sum()
        rng = self.random_state

        match self.bootstrap_type:
            case 'frequentist':
                if self.full_model is None:
                    msg = "Bootstrap type 'frequentist' requires a fitted \
                        model."
                    logger.critical(msg)
                    raise ValueError(msg)

                logger.debug("shuffling data-model differences.")
                # Shuffle data-model differences
                m = self.full_model(x)
                diff = y - m
                def f() -> FloatVector:
                    return m + rng.choice(diff, size=n_points, replace=True)

            case 'model':
                if self.full_model is None:
                    msg = "Bootstrap type 'model' requires a fitted model."
                    logger.critical(msg)
                    raise ValueError(msg)

                logger.debug("adding noise to fitted model.")
                # Add noise to fitted model
                m = self.full_model(x)
                def f() -> FloatVector:
                    return rng.normal(loc=m, scale=dy)

            case _:
                logger.debug("adding noise to observed spectrum.")
                # Add noise to spectrum
                def f() -> FloatVector:
                    return rng.normal(loc=y, scale=dy)

        return (f() for _ in range(self.iterations))

    def run(self) -> list[Sample]:
        """
        Runs the bootstrapping procedure, returning a list of 'Sample' objects
        generated from the resampled flux density arrays.
        """
        if self.tqdm_disable:
            loop = self.generateSamples()
        else:
            loop = tqdm(
                self.generateSamples(), 
                total=self.iterations, 
                desc=f"Bootstrapping ({self.variant})",
                unit='sample',
                leave=self.tqdm_leave,
            )
        return list(self.mapper(self.mapped_func, loop))

    @validate_call
    def toErrorResult(
        self,
        samples: Iterable[Sample] | None = None,
    ) -> ErrorResult:
        """
        ** PYDANTIC VALIDATED METHOD **

        Builds and returns an 'ErrorResult' object from the original sample and
        any additional samples provided as an argument.
        """
        result = ErrorResultBuilder()
        result.simple_append(self.base_sample)

        if samples is not None:
            result.simple_extend(samples)

        return result.finalise()