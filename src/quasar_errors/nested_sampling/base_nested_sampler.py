from typing import Self, Iterable, Literal, ClassVar, Callable
from logging import getLogger
from numpy import isfinite, invert, argwhere, median, exp, empty, float64
from numpy.random import default_rng
from dataclasses import field
from pydantic.dataclasses import dataclass

from quasar_typing.numpy import FloatVector, BoolVector, RandomState_, FittableCoordsTuple
from quasar_typing.astropy import Model_, CompoundModel_
from quasar_typing.dynesty import NestedSampler_, DynamicNestedSampler_
from quasar_typing.misc import Pool_

from quasar_utils.setup import Info
from quasar_utils.decorators import apply_info_to_method, validate_call

from quasar_models import PowerLawModel, IronModel, BalmerModel, GaussianModel
from quasar_models.line import _VProfileCopy

from ..model_samples import Sample
from .pseudo_compound_model import PseudoCompoundModel

from .loss_functions import chi2
from .prior_transforms import uniform

from ..error_result import ErrorResult, ErrorResultBuilder

from dynesty import NestedSampler, DynamicNestedSampler

logger = getLogger(__name__)

@dataclass
class BaseNestedSampler:
    x: FloatVector
    y: FloatVector
    dy: FloatVector

    usable_pixels: BoolVector | None = None
    info: Info = field(default_factory=Info)

    pl: PowerLawModel | None = None
    fe: IronModel | CompoundModel_[IronModel] | None = None
    ba: BalmerModel | None = None
    hg: Model_ | None = None
    em: GaussianModel | _VProfileCopy | CompoundModel_[GaussianModel | _VProfileCopy] | None = None

    pool: Pool_ | None = None
    iterations: int | None = None
    random_state: RandomState_ | None = None
    renew_rng: bool | None = None
    replace_missing: bool | None = None
    tqdm_disable: bool | None = None
    tqdm_leave: bool | None = None

    # Set by __post_init__
    random_seed: dict = field(init=False)
    full_model: Model_ = field(init=False)
    base_sample: Sample = field(init=False)
    pseudo_model: PseudoCompoundModel = field(init=False)
    data: FittableCoordsTuple = field(init=False)
    prior_transform: Callable[[FloatVector, PseudoCompoundModel], FloatVector] = field(init=False)
    loglikelihood: Callable[[FloatVector, PseudoCompoundModel], float] = field(init=False)
    normalisation: float = field(init=False)

    sampler: NestedSampler_ | DynamicNestedSampler_ | None = field(default=None, init=False, repr=False)

    variant: Literal['standard'] = field(default='standard', init=False)

    prior: ClassVar[Literal['uniform']]
    loss: ClassVar[Literal['chi2']]

    def __post_init__(self):
        if self.usable_pixels is None:
            self.usable_pixels = (
                isfinite(self.x) & isfinite(self.y) & isfinite(self.dy)
                & (self.dy > 0)
            )

        self.setHyperParams(
            iterations=self.iterations,
            random_state=self.random_state,
            renew_rng=self.renew_rng,
            replace_missing=self.replace_missing,
            tqdm_disable=self.tqdm_disable,
            tqdm_leave=self.tqdm_leave,
        )

        print(self.random_state, type(self.random_state))
        self.random_seed = self.random_state.get_state()

        if self.replace_missing and not self.usable_pixels.any():
            self.replaceMissing()

        self._set_data()
        self._set_full_model()
        self._set_base_sample()
        self._set_pseudo_model()
        self._set_prior_transform()
        self._set_loglikelihood()

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
            self.variant, x.size,
        )
        self.base_sample = Sample.fromModelSample(x, y, dy, model=self.full_model)
    
    def _set_pseudo_model(self) -> None:
        """
        Sets the 'PseudoCompoundModel' representation of the full model.
        """
        self.pseudo_model = PseudoCompoundModel.from_model(self.full_model)
    
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
        match self.prior:
            case 'uniform':
                self.prior_transform = uniform.prior_transform
            case _:
                raise ValueError(f"Unsupported prior: {self.prior}")
            
    def _set_loglikelihood(self) -> None:
        """
        Sets the log-likelihood function based on the specified loss.
        """
        match self.loss:
            case 'chi2':
                self.loglikelihood = chi2.loglike
                self.normalisation = chi2.normalisation(self.data[2])
            case _:
                raise ValueError(f"Unsupported loss: {self.loss}")
        
    @apply_info_to_method(subjects=('error',))
    def setHyperParams(
        self,
        *,
        iterations: int | None = None,
        random_state: RandomState_ | None = None,
        renew_rng: bool | None = None,
        replace_missing: bool | None = None,
        tqdm_disable: bool | None = None,
        tqdm_leave: bool | None = None,
    ) -> None:
        self.iterations = iterations
        self.random_state = random_state
        self.renew_rng = renew_rng
        self.replace_missing = replace_missing
        self.tqdm_disable = tqdm_disable
        self.tqdm_leave = tqdm_leave

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
            invert(self.usable_pixels).sum(), len(self.usable_pixels),
        )
        if self.usable_pixels.all():
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

            self.x[invalid_x] = self.x[first] * (1 + self.info.loading.sigma_res)**powers

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

        # Make every element True.
        self.usable_pixels[:] = True
        msg += f"complete! All {self.usable_pixels.size} pixel(s) are usable"
        logger.debug(msg)

        return self
    
    def instantiateSampler(
        self,
        *,
        dynamic: bool = True,
        dynesty_bound: str = 'multi', 
        dynesty_sample: str = 'auto',
    ) -> NestedSampler_ | DynamicNestedSampler_:
        """
        Runs the nested sampler, returning a list of 'Sample' objects.
        """
        logl_kwargs = {
            'model': self.pseudo_model,
            'data': (
                self.x[self.usable_pixels], 
                self.y[self.usable_pixels], 
                self.dy[self.usable_pixels],
            ),
            'normalisation': self.normalisation,
        }
        ptform_kwargs = {'model': self.pseudo_model}

        SamplerCls = DynamicNestedSampler if dynamic else NestedSampler()
        self.sampler = SamplerCls(
            self.loglikelihood,
            self.prior_transform,
            self.pseudo_model.n_free,
            bound=dynesty_bound,
            sample=dynesty_sample,
            rstate=default_rng(self.random_state),
            logl_kwargs=logl_kwargs,
            ptform_kwargs=ptform_kwargs,
        )
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

        kwargs = {'print_progress': print_progress}
        if isinstance(self.sampler, DynamicNestedSampler):
            kwargs['maxiter_init'] = maxiter
            kwargs['dlogz_init'] = dlogz
        else:
            kwargs['maxiter'] = maxiter
            kwargs['dlogz'] = dlogz


        self.sampler.run_nested(**kwargs)
        results = self.sampler.results
        weights = exp(results.logwt - results.logwt.max())

        chosen_indices = self.random_state.choice(
            results.niter,
            size=self.iterations,
            replace=True,
            p=weights / weights.sum(),
        )
        x = self.x[self.usable_pixels]
        y = self.y[self.usable_pixels]
        dy = self.dy[self.usable_pixels]

        samples: list[Sample] = []
        for idx in enumerate(chosen_indices):
            param_vector = self.prior_transform(
                results.samples[idx],
                self.pseudo_model,
            )
            samples.append(Sample.fromParamVector(
                x, y, dy,
                param_vector,
                self.full_model,
                weight=1.0,
            ))

        return samples

    @validate_call
    def toErrorResult(
        self,
        samples: Iterable[Sample] | None = None,
    ) -> ErrorResult:
        result = ErrorResultBuilder()
        result.simple_append(self.base_sample)

        if samples:
            result.simple_extend(samples)

        return result.finalise()