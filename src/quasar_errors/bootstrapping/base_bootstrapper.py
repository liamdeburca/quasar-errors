import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import field
from typing import ClassVar, Optional, Self, Union

import numpy as np
import pydantic
from quasar_models import (
    BalmerModel,
    EmModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.modeling import PrepareModel
from quasar_models.modeling.base_model import BaseModel
from quasar_typing.astropy import CompoundModelType, ModelType
from quasar_typing.literals import BootstrapType, Variant
from quasar_typing.multiprocessing import PoolType
from quasar_typing.numpy import (
    BoolVector,
    FloatMatrix,
    FloatVector,
    RandomStateType,
)
from quasar_typing.string_selections import VaryLines
from quasar_utils.continuum_fit_result import ContinuumFitResult
from quasar_utils.decorators import apply_info_to_method
from quasar_utils.setup import FitterKwargs, Info

from quasar_errors.bootstrapping.mapped_func import MappedFunc
from quasar_errors.error_result import ErrorResult, ErrorResultBuilder
from quasar_errors.model_samples import Sample

logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(repr=False)
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
    dx: FloatVector

    kernels: FloatMatrix | None = None
    eval_mask: BoolVector | None = None
    comp_mask: BoolVector | None = None

    is_logbinned: bool | None = None
    info: Info = None

    cfit: ContinuumFitResult | None = None
    pl: Optional[PowerLawModel] = None
    fe: Union[IronModel, CompoundModelType[IronModel], None] = None
    ba: Optional[BalmerModel] = None
    hg: Optional[HostGalaxyModel] = None
    em: Union[EmModel, CompoundModelType[EmModel], None] = None

    pool: PoolType | None = None

    # Hyperparameters
    fitter_kwargs: FitterKwargs | None = None
    iterations: int | None = None
    random_state: RandomStateType | None = None
    bootstrap_type: BootstrapType | None = None
    n_sigmas: float | None = None
    replace_missing: bool | None = None
    vary_lines: VaryLines | None = None
    tqdm_disable: bool | None = None
    tqdm_leave: bool | None = None

    # Set by __post_init__
    y_bg: FloatVector = field(init=False)
    full_model: Optional[ModelType] = field(init=False)
    fitted_model: Optional[ModelType] = field(init=False)
    mapped_func: MappedFunc = field(init=False)
    mapper: Callable[[MappedFunc, Iterable[FloatVector]], Iterator[Sample]] = (
        field(init=False)
    )

    def __repr__(self) -> str:
        model_types = {
            t
            for t in ["pl", "fe", "ba", "hg", "em"]
            if getattr(self, t) is not None
        }
        
        s = f"<{self.__class__.__name__} at {hex(id(self))}> "
        s += f"type={self.bootstrap_type}, {model_types=}"

        return s

    def __post_init__(self):
        logger.debug("BaseBootstrapper.__post_init__():")

        if self.eval_mask is None:
            logger.debug(">>> Setting `eval_mask` from data.")
            self.eval_mask = np.isfinite(self.x)
        elif self.eval_mask.shape != self.x.shape:
            msg = f"Shape mismatch between `eval_mask` {self.eval_mask.shape} and `x` {self.x.shape}."
            logger.critical(">>>" + msg)
            raise ValueError(msg)

        if self.comp_mask is None:
            logger.debug(">>> Setting `comp_mask` from data.")
            self.comp_mask = np.isfinite([self.y, self.dy]).all(axis=0) & (self.dy > 0)
        elif self.comp_mask.shape != self.x.shape:
            msg = f"Shape mismatch between `comp_mask` {self.comp_mask.shape} and `x` {self.x.shape}."
            logger.critical(">>>" + msg)
            raise ValueError(msg)

        logger.debug(">>> Setting hyperparameters.")
        self.setHyperParams(
            fitter_kwargs=self.fitter_kwargs,
            iterations=self.iterations,
            random_state=self.random_state,
            bootstrap_type=self.bootstrap_type,
            replace_missing=self.replace_missing,
            vary_lines=self.vary_lines,
            tqdm_disable=self.tqdm_disable,
            tqdm_leave=self.tqdm_leave,
        )

        logger.debug(">>> Initializing background array.")
        self.y_bg = np.zeros_like(self.x, np.float64, "C")
        
        logger.debug(">>> Computing full model.")
        self._compute_full_model()

        if self.replace_missing and not self.comp_mask[self.eval_mask].all():
            logger.debug(">>> Replacing missing data.")
            self.replaceMissing()

        logger.debug(">>> Computing fitted model.")
        self._compute_fitted_model()

        logger.debug(">>> Computing mapped function.")
        self._compute_mapped_func()

        if self.pool is None:
            self.mapper = map
        elif self.tqdm_disable:
            self.mapper = self.pool.map
        else:
            self.mapper = self.pool.imap

        mask = self.comp_mask & self.eval_mask
        logger.debug(
            ">>> %d/%d usable pixels, bootstrap_type=%r, iterations=%d.",
            mask.sum(),
            mask.size,
            self.bootstrap_type,
            self.iterations,
        )

    def _compute_full_model(self) -> Optional[ModelType]:
        """
        Full model, i.e. the sum of all available model components.
        If none are available, returns None.
        """
        submodels = []
        for component in filter(
            None, 
            (self.pl, self.fe, self.ba, self.hg, self.em),
        ):
            if isinstance(component, BaseModel):
                submodels.append(component)
            else:
                submodels.extend(comp for comp in component)

        if not submodels:
            logger.debug("no model components provided, returning None.")
            return None

        submodels = sorted(submodels, key=lambda m: m.sorting_key)
        self.full_model = sum(submodels[1:], start=submodels[0])

        logger.debug("built from %d submodel(s).", len(submodels))
        return self.full_model

    def _compute_fitted_model(self) -> Optional[ModelType]:
        """
        The model to be refitted at each iteration.
        """
        filter_func = lambda m: m is not None
        match self.variant:
            case "spectrum":
                logger.debug("variant=%r, no refitting.", self.variant)
                bgs = ()
                fitted_model = None
            case "standard":
                if self.full_model is None:
                    msg = "Bootstrapping variant 'standard' requires a fitted "\
                        "model."
                    logger.critical(msg)
                    raise ValueError(msg)

                logger.debug("variant=%r, refitting full model.", self.variant)
                bgs = ()
                fitted_model = self.full_model
            case "flexible":
                if self.pl is None or self.em is None:
                    msg = "Bootstrapping variant 'flexible' requires both a "\
                        "power law and an emission line model. Missing: "\
                        f"pl={self.pl is None}, em={self.em is None}"
                    logger.critical(msg)
                    raise ValueError(msg)
                if self.cfit is None:
                    msg = "Bootstrapping variant 'flexible' requires a "\
                        "continuum fit result."
                    logger.critical(msg)
                    raise ValueError(msg)

                pl = self.pl.copy()

                s = self.cfit.std[0]
                pl.flux.bounds = (
                    max(
                        pl.flux.bounds[0], 
                        pl.flux.value - self.n_sigmas * s,
                    ),
                    min(
                        pl.flux.bounds[1], 
                        pl.flux.value + self.n_sigmas * s,
                    ),
                )
                s = self.cfit.std[1]
                pl.alpha.bounds = (
                    max(
                        pl.alpha.bounds[0], 
                        pl.alpha.value - self.n_sigmas * s
                    ),
                    min(
                        pl.alpha.bounds[1], 
                        pl.alpha.value + self.n_sigmas * s
                    ),
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
            case "rigid":
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
                        self.variant,
                        len(submodels),
                        len(submodels) + len(extra_bgs),
                        self.vary_lines,
                    )

        mask = self.comp_mask & self.eval_mask
        x = self.x[mask]
        self.y_bg[:] = 0.0
        self.y_bg[mask] = sum((f(x) for f in bgs), start=0.0)

        self.fitted_model = fitted_model

        return self.fitted_model

    def _compute_mapped_func(self) -> MappedFunc:
        """
        Returns the function to be mapped over the generated samples, i.e. the
        function that creates a 'Sample' object from a resampled flux density
        array.
        """
        kwargs = {"variant": self.variant}
        mask = self.comp_mask & self.eval_mask
        if self.variant == "spectrum":
            logger.debug(
                "variant=%r, npix=%d/%d, building function w/o fitting",
                self.variant,
                mask.sum(),
                mask.size,
            )
        else:
            logger.debug(
                "variant=%r, npix=%d/%d, building function w/ fitting",
                self.variant,
                mask.sum(),
                mask.size,
            )
            kwargs["model"] = self.fitted_model.copy()
            kwargs["fitter_kwargs"] = self.fitter_kwargs
            kwargs["kernels"] = self.kernels

        kwargs["is_logbinned"] = self.is_logbinned
        kwargs["eval_mask"] = self.eval_mask
        kwargs["comp_mask"] = self.comp_mask

        self.mapped_func = MappedFunc(
            self.x, self.dy, self.dx, self.y_bg, 
            **kwargs,
        )
        return self.mapped_func

    def initializeResult(self) -> ErrorResultBuilder:
        """
        Corresponding 'Sample' object for the original data.
        """
        mask = self.comp_mask & self.eval_mask
        x = self.x[mask]
        y = self.y[mask]
        dy = self.dy[mask]
        dx = self.dx[mask]

        if self.full_model is None:
            logger.debug(
                "creating base sample w/o model; variant=%r, npix=%d/%d.",
                self.variant,
                mask.sum(),
                mask.size,
            )
            base_sample = Sample.fromSpectrumSample(
                x, y, dy, dx, 
                weight=1.0,
            )
            seq_model = None
        else:
            logger.debug(
                "creating base sample w/ model; variant=%r, npix=%d/%d.",
                self.variant,
                mask.sum(),
                mask.size,
            )
            preparer = PrepareModel(
                x=self.x,
                model=self.full_model,
                eval_mask=self.eval_mask,
                comp_mask=self.comp_mask,
                kernels=self.kernels,
                is_logbinned=self.is_logbinned,
            )
            with preparer as (seq_model, _):
                base_sample = Sample.fromModelSample(
                    x, y, dy, dx, 
                    model=seq_model,
                    weight=1.0,
                )
        
        return ErrorResultBuilder(
            seq_model=seq_model,
            samples=[base_sample],
            sample_weights=[1.0],
        )


    @apply_info_to_method(subjects=("nonlinear", "error", "random"))
    def setHyperParams(
        self,
        *,
        fitter_kwargs: FitterKwargs | None = None,
        iterations: int | None = None,
        bootstrap_type: BootstrapType | None = None,
        replace_missing: bool | None = None,
        vary_lines: VaryLines | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
        random_state: RandomStateType | None = None,
    ) -> None:
        """
        ** INFO APPLIED TO METHOD **

        Resolves and assigns hyperparameter fields from the provided values,
        falling back to defaults drawn from 'self.info' via
        @apply_info_to_method for any argument left as None:
        """
        self.fitter_kwargs = fitter_kwargs
        self.iterations = iterations
        self.random_state = random_state
        self.bootstrap_type = bootstrap_type
        self.replace_missing = replace_missing
        self.vary_lines = vary_lines
        self.tqdm_disable = tqdm_disable
        self.tqdm_leave = tqdm_leave
        logger.debug(
            f"{fitter_kwargs=}, {iterations=}, {random_state=}, "
            f"{bootstrap_type=}, {replace_missing=}, {vary_lines=}, "
            f"{tqdm_disable=}, {tqdm_leave=}.",
        )

    def replaceMissing(self) -> Self:
        """
        ...

        Notes
        -----
        If 'self.bootstrap_type' is 'frequentist', i.e. the data-model
        differences are shuffled at each iteration, this method does nothing.

        In the event that 'self.bootstrap_type' is 'spectrum', the pixels
        lacking both valid flux density and flux density uncertainty values,
        will have their replaced uncertainties amplified by a factor of
        sqrt(2).
        """
        from .helpers import replace_missing_helper

        logger.debug("BaseBootstrapper.replaceMissing():")
        msgs = replace_missing_helper(
            x=self.x, 
            y=self.y, 
            dy=self.dy,
            eval_mask=self.eval_mask, 
            comp_mask=self.comp_mask,
            full_model=self.full_model,
            kernels=self.kernels,
            is_logbinned=self.is_logbinned,
            bootstrap_type=self.bootstrap_type,
        )
        for msg in msgs:
            logger.debug(">>> " + msg)

        return self

    def generateSamples(self) -> Iterator[FloatVector]:
        """
        Creates an iterator that generates resampled flux density arrays
        according to 'self.bootstrap_type' and the available model components.
        """
        from .helpers import generate_samples_helper

        logger.debug("BaseBootstrapper.generateSamples():")
        msgs, samples_iter = generate_samples_helper(
            x=self.x, 
            y=self.y, 
            dy=self.dy,
            eval_mask=self.eval_mask, 
            comp_mask=self.comp_mask,
            full_model=self.full_model,
            kernels=self.kernels,
            is_logbinned=self.is_logbinned,
            bootstrap_type=self.bootstrap_type,
            iterations=self.iterations,
            rng=self.random_state,
        )
        for msg in msgs:
            logger.debug(">>> " + msg)

        return samples_iter

    def run(self) -> list[Sample]:
        """
        Runs the bootstrapping procedure, returning a list of 'Sample' objects
        generated from the resampled flux density arrays.
        """
        from tqdm import tqdm

        with self.mapped_func as func:
            samples = list(
                tqdm(
                    self.mapper(func, self.generateSamples()),
                    total=self.iterations,
                    desc=f"Bootstrapping ({self.variant})",
                    unit="sample",
                    leave=self.tqdm_leave,
                    disable=self.tqdm_disable,
                )
            )
        return samples

    @pydantic.validate_call
    def toErrorResult(
        self,
        samples: Iterable[Sample] | None = None,
    ) -> ErrorResult:
        """
        Builds and returns an 'ErrorResult' object from the original sample and
        any additional samples provided as an argument.
        """
        result = self.initializeResult()
        if samples is not None:
            result.simple_extend(samples)
        return result.finalise()
