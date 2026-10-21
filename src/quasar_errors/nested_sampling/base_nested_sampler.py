from collections.abc import Callable, Iterable
from dataclasses import field
from logging import getLogger
from typing import ClassVar, Literal, Optional, Protocol, Self, Union

import numpy as np
import pydantic
from dynesty import DynamicNestedSampler, NestedSampler
from numpy.random import default_rng
from quasar_models import (
    BalmerModel,
    EmModel,
    HostGalaxyModel,
    IronModel,
    PowerLawModel,
)
from quasar_models.modeling import SequentialModel
from quasar_typing.astropy import CompoundModelType, ModelType
from quasar_typing.dynesty import DynamicNestedSamplerType, SamplerType
from quasar_typing.literals import Variant
from quasar_typing.logging import LoggerType
from quasar_typing.multiprocessing import PoolType
from quasar_typing.numpy import (
    BoolVector,
    FittableCoordsTuple,
    FloatVector,
    RandomStateType,
)
from quasar_typing.string_selections import (
    DataTypes,
    ModelTypes,
)
from quasar_utils.decorators import apply_info_to_method
from quasar_utils.setup import Info

from quasar_errors.error_result import ErrorResult, ErrorResultBuilder
from quasar_errors.model_samples import Sample

logger = getLogger(__name__)


class SpectrumProtocol(Protocol):
    _x: FloatVector
    _y: FloatVector
    _dy: FloatVector
    info: Info


@pydantic.dataclasses.dataclass
class BaseNestedSampler:
    x: FloatVector = field(kw_only=True)
    y: FloatVector = field(kw_only=True)
    dy: FloatVector = field(kw_only=True)

    usable_pixels: BoolVector | None = field(default=None, kw_only=True)
    info: Info = field(default_factory=Info, kw_only=True)

    pl: Optional[PowerLawModel] = field(
        default=None, 
        kw_only=True,
    )
    fe: Union[IronModel, CompoundModelType[IronModel], None] = field(
        default=None, 
        kw_only=True,
    )
    ba: Optional[BalmerModel] = field(
        default=None, 
        kw_only=True,
    )
    hg: Optional[HostGalaxyModel] = field(
        default=None, 
        kw_only=True,
    )
    em: Union[EmModel, CompoundModelType[EmModel], None] = field(
        default=None, 
        kw_only=True,
    )

    pool: PoolType | None = field(default=None, kw_only=True)
    iterations: int | None = field(default=None, kw_only=True)
    random_state: RandomStateType | None = field(default=None, kw_only=True)
    replace_missing: bool | None = field(default=None, kw_only=True)
    tqdm_disable: bool | None = field(default=None, kw_only=True)
    tqdm_leave: bool | None = field(default=None, kw_only=True)

    # Set by __post_init__
    random_seed: dict = field(init=False)
    full_model: ModelType = field(init=False)
    base_sample: Sample = field(init=False)
    seq_model: SequentialModel = field(init=False)
    data: FittableCoordsTuple = field(init=False)
    prior_transform: Callable[[FloatVector, FloatVector, FloatVector], FloatVector] = field(init=False)
    loglikelihood: Callable[[FloatVector, SequentialModel], float] = field(init=False)
    normalisation: float = field(init=False)

    sampler: SamplerType | DynamicNestedSamplerType | None = field(default=None, init=False, repr=False)

    variant: Literal["standard"] = field(default="standard", init=False)

    prior: ClassVar[Literal["uniform"]]
    loss: ClassVar[Literal["chi2"]]

    def prepare(self) -> None:
        if self.usable_pixels is None:
            self.usable_pixels = np.isfinite(
                [self.x, self.y, self.dy]
            ).all(axis=0) & (self.dy > 0)

        self.setHyperParams(
            iterations=self.iterations,
            random_state=self.random_state,
            replace_missing=self.replace_missing,
            tqdm_disable=self.tqdm_disable,
            tqdm_leave=self.tqdm_leave,
        )
        self.random_seed = self.random_state.get_state()

        if self.replace_missing and not self.usable_pixels.any():
            self.replaceMissing()

        self._set_data()
        self._set_full_model()
        self._set_base_sample()
        self._set_pseudo_model()
        self._set_prior_transform()
        self._set_loglikelihood()

    @classmethod
    def from_spectrum(
        cls,
        *,
        spectrum: SpectrumProtocol,
        covered: bool = False,
        without_rejections: bool = False,
        without_absorption: bool = False,
        variant: Variant = "standard",
        data_types: DataTypes | None = None,
        model_types: ModelTypes | None = None,
        pool: PoolType | None = None,
        iterations: int = 100,
        random_state: RandomStateType = None,
        replace_missing: bool = True,
        tqdm_disable: bool = False,
        tqdm_leave: bool = False,
        logger: LoggerType = logger,
    ) -> Self:
        if model_types is None:
            model_types = ModelTypes({'all'})

        kwargs = {
            "x": spectrum._x,
            "y": spectrum._y,
            "dy": spectrum._dy,
            "info": spectrum.info,
            "pool": pool,
            "iterations": iterations,
            "random_state": random_state,
            "replace_missing": replace_missing,
            "tqdm_disable": tqdm_disable,
            "tqdm_leave": tqdm_leave,
        }

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
                msg = (
                    f"Could not find '{r_type}' windows required for '{variant}' nested "
                    "sampling! Cancelling nested sampling."
                )
                logger.critical(msg)
                raise ValueError(msg)
            elif spectrum[r_type].getModel() is None:
                msg = (
                    f"Could not find '{r_type}' model required for '{variant}' nested "
                    "sampling! Cancelling nested sampling."
                )
                logger.critical(msg)
                raise ValueError(msg)

        for m_type in model_types:
            if (windows := spectrum[m_type]) is None or (model := windows.getModel()) is None:
                continue
            kwargs[m_type] = model

        ### Data types // Windows ###
        if data_types is not None:
            for d_type in filter(
                lambda d_type: d_type not in model_types, data_types
            ):
                logger.warning(
                    f"Required model type '{d_type}' is not included in "
                    "'data_types'! This may lead to inaccurate error estimation."
                )

        kwargs["usable_pixels"] = spectrum.getMask.__wrapped__(
            spectrum,
            data_types=data_types,
            covered=covered,
            without_rejections=without_rejections,
            without_absorption=without_absorption,
            valid=True,
        )
        nested_sampler = cls(**kwargs)
        nested_sampler.prepare()
        return nested_sampler

    def _set_full_model(self) -> None:
        submodels = []
        for component in (self.pl, self.fe, self.ba, self.hg, self.em):
            if component is None:
                continue
            elif isinstance(component, Iterable):
                submodels.extend(component)
            else:
                submodels.append(component)

        assert len(submodels) >= 1, "At least one component must be provided."

        self.full_model = sum(submodels[1:], start=submodels[0])

    def _set_base_sample(self) -> None:
        """
        Sets the corresponding 'Sample' object for the original data.
        """
        x, y, dy = self.data
        logger.debug(
            "creating base sample w/ model; variant=%r, npix=%d.",
            self.variant,
            x.size,
        )
        self.base_sample = Sample.fromModelSample(
            x, y, dy, model=self.full_model
        )

    def _set_sequential_model(self) -> None:
        """
        Sets the 'SequentialModel'.
        """
        self.seq_model = SequentialModel(self.full_model)

    def _set_data(self) -> None:
        """
        Sets the data as a tuple of (x, y, dy) for the usable pixels.
        """
        self.data = (
            self.x[self.usable_pixels],
            self.y[self.usable_pixels],
            self.dy[self.usable_pixels],
        )

    def _set_prior_transform(self) -> None:
        """
        Sets the prior transform function based on the specified prior.
        """
        from .prior_transforms import uniform
        match self.prior:
            case "uniform":
                self.prior_transform = uniform.prior_transform
            case _:
                raise ValueError(f"Unsupported prior: {self.prior}")

    def _set_loglikelihood(self) -> None:
        """
        Sets the log-likelihood function based on the specified loss.
        """
        from .loss_functions import chi2
        match self.loss:
            case "chi2":
                self.loglikelihood = chi2.loglike
                self.normalisation = chi2.normalisation(self.data[2])
            case _:
                raise ValueError(f"Unsupported loss: {self.loss}")

    @apply_info_to_method(subjects=("error", "random"))
    def setHyperParams(
        self,
        *,
        iterations: int | None = None,
        replace_missing: bool | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
        random_state: RandomStateType | None = None,
    ) -> None:
        self.iterations = iterations
        self.replace_missing = replace_missing
        self.tqdm_disable = tqdm_disable
        self.tqdm_leave = tqdm_leave
        self.random_state = random_state

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
        """
        logger.debug(
            "%d/%d non-usable pixel(s).",
            (~self.usable_pixels).sum(),
            len(self.usable_pixels),
        )
        if self.usable_pixels.all():
            return self

        msg = ""

        valid_x = np.isfinite(self.x)
        invalid_x = ~valid_x
        if valid_x.all():
            msg += "no missing 'x' values, "
        else:
            msg += f"replacing {invalid_x.sum()} missing 'x' value(s), "
            valid_indices = np.argwhere(valid_x).flatten()
            invalid_indices = np.argwhere(invalid_x).flatten()

            first = valid_indices[0]
            powers = invalid_indices - first

            self.x[invalid_x] = (
                self.x[first] * (1 + self.info.loading.sigma_res) ** powers
            )

        valid_y = np.isfinite(self.y) & self.usable_pixels
        invalid_y = ~valid_y
        if valid_y.all():
            msg += "no missing 'y' values, "
        else:
            msg += f"replacing {invalid_y.sum()} missing 'y' value(s), "
            self.y[invalid_y] = self.full_model(self.x[invalid_y])

        valid_dy = np.isfinite(self.dy) & (self.dy > 0)
        invalid_dy = ~valid_dy
        if valid_dy.all():
            msg += "no missing 'dy' values."
        else:
            msg += f"replacing {invalid_dy.sum()} missing 'dy' value(s), "
            self.dy[invalid_dy] = np.median(self.dy[valid_dy])

        # Make every element True.
        self.usable_pixels[:] = True
        msg += f"complete! All {self.usable_pixels.size} pixel(s) are usable"
        logger.debug(msg)

        return self

    def instantiateSampler(
        self,
        *,
        dynamic: bool = True,
        dynesty_bound: str = "multi",
        dynesty_sample: str = "auto",
    ) -> SamplerType | DynamicNestedSamplerType:
        """
        Runs the nested sampler, returning a list of 'Sample' objects.
        """
        logl_kwargs = {
            "model": self.seq_model,
            "data": (
                np.ascontiguousarray(self.x[self.usable_pixels]),
                np.ascontiguousarray(self.y[self.usable_pixels]),
                np.ascontiguousarray(self.dy[self.usable_pixels]),
            ),
            "normalisation": self.normalisation,
        }
        ptform_kwargs = {
            "lbs": (bounds := self.seq_model.bounds)[0],
            "ubs": bounds[1],
        }

        args = (
            self.loglikelihood,
            self.prior_transform,
            self.seq_model._n_free,
        )
        kwargs = {
            "bound": dynesty_bound,
            "sample": dynesty_sample,
            "rstate": default_rng(self.random_state),
            "logl_kwargs": logl_kwargs,
            "ptform_kwargs": ptform_kwargs,
        }
        if dynamic:
            self.sampler = DynamicNestedSampler(*args, **kwargs)
        else:
            self.sampler = NestedSampler(*args, **kwargs)
        return self.sampler

    def run(
        self,
        *,
        maxiter: int | None = None,
        dlogz: float | None = None,
        print_progress: bool | None = None,
    ) -> list[Sample]:
        """
        Runs the nested sampler, returning a list of 'Sample' objects.
        """
        assert self.sampler is not None

        kwargs = {"print_progress": print_progress}
        if isinstance(self.sampler, DynamicNestedSampler):
            kwargs["maxiter_init"] = maxiter
            kwargs["dlogz_init"] = dlogz
        else:
            kwargs["maxiter"] = maxiter
            kwargs["dlogz"] = dlogz

        self.sampler.run_nested(**kwargs)
        results = self.sampler.results
        weights = np.exp(results.logwt - results.logwt.max())

        chosen_indices = self.random_state.choice(
            results.niter,
            size=self.iterations,
            replace=True,
            p=weights / weights.sum(),
        )
        x = self.x[self.usable_pixels]
        y = self.y[self.usable_pixels]
        dy = self.dy[self.usable_pixels]

        return [
            Sample.fromParamVector(
                x, y, dy,
                self.seq_model._get_full_parameter_array(params),
                self.full_model,
                weight=1.0,
            ) 
            for params in results.samples[chosen_indices]
        ]

    @pydantic.validate_call
    def toErrorResult(
        self,
        samples: Iterable[Sample] | None = None,
    ) -> ErrorResult:
        result = ErrorResultBuilder()
        result.simple_append(self.base_sample)

        if samples:
            result.simple_extend(samples)

        return result.finalise()
