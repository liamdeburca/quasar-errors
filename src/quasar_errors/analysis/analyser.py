from collections.abc import Callable, Iterable, Iterator
from dataclasses import field
from functools import cached_property
from itertools import repeat, starmap
from logging import getLogger
from typing import ClassVar, Literal

from numpy import (
    allclose,
    array,
    dot,
    einsum,
    empty,
    float64,
    full,
    hypot,
    inf,
    invert,
    linspace,
    ones,
    stack,
    where,
)
from pydantic.dataclasses import dataclass
from quasar_typing.misc import BackgroundFlux, FWHMStrategy, Pool_
from quasar_typing.numpy import BoolMatrix, FloatMatrix, FloatVector
from quasar_utils.decorators import validate_call, validated_lru_cache
from quasar_utils.setup import Info

from ..error_result import ErrorResult
from ..measures import approx, exact
from . import inner, metadata, spectrum_inner
from .discretised_coords import DiscretisedCoords
from .measure_result import MeasureResult
from .moment_params import MomentParams

logger = getLogger(__name__)


@dataclass(repr=False)
class Analyser:
    error_result: ErrorResult

    wave: float | None = None
    line: str | None = None

    pool: Pool_ | None = None
    info: Info = None
    res: int | None = None
    render_width: float | None = None
    fwhm_strategy: FWHMStrategy | None = None
    exact: bool | None = None
    v_int: float | None = None
    ipv_int: float | None = None
    dx_int: float | None = None

    sample_weights: FloatVector = field(init=False)
    x: FloatVector = field(init=False)
    y: FloatVector = field(init=False)
    dx: FloatVector = field(init=False)
    dy: FloatVector = field(init=False)

    methods: ClassVar[frozenset[str]] = metadata.methods
    measures: ClassVar[frozenset[str]] = metadata.measures

    def __post_init__(self):
        if (self.wave is not None) and (self.line is not None):
            self.wave = None
            logger.warning(
                "Only filtering for a single line or wave is supported! "
                "Setting 'wave' to None."
            )

        if self.res is None:
            self.res = self.info.error.res
        if self.render_width is None:
            self.render_width = self.info.error.render_width
        if self.fwhm_strategy is None:
            self.fwhm_strategy = self.info.error.fwhm_strategy
        if self.exact is None:
            self.exact = self.info.error.exact
        if self.v_int is None:
            self.v_int = self.info.error.v_int
        if self.ipv_int is None:
            self.ipv_int = self.info.error.ipv_int
        if self.dx_int is None:
            self.dx_int = self.info.error.dx_int

        self.error_result = self.error_result.isolate.__wrapped__(
            self.error_result,
            target_line=self.line,
            target_wave=self.wave,
        )
        if self.error_result[0].em is None:
            logger.warning(
                "Isolated error result has no emission line model! "
                "Do not perform emission line measurements to avoid errors."
            )
        else:
            self.wave = self.error_result[0].em[0].wave
            self.line = self.error_result[0].em[0].pure_name

        self.sample_weights = array(self.error_result.sample_weights)
        self.sample_weights /= self.sample_weights.sum()

        self.x: FloatVector = self.error_result[0].x
        self.y: FloatVector = self.error_result[0].y
        self.dy: FloatVector = self.error_result[0].dy
        self.dx: FloatVector = self.x * self.info.loading.sigma_res

        self.n_samples: int = self.error_result.n_samples
        self.n_pix: int = self.error_result.n_pix

    def __repr__(self) -> str:
        _result = hex(id(self))
        return f"<Analyser {_result} for {self.line} w/ {self.n_samples} samples >"

    def __hash__(self) -> str:
        return id(self)

    @property
    def map(self) -> Callable[[Callable, Iterable], Iterator]:
        if self.pool is not None:
            return self.pool.map
        return map

    @property
    def starmap(self) -> Callable[[Callable, Iterable], Iterator]:
        if self.pool is not None:
            return self.pool.starmap
        return starmap

    @cached_property
    def y_pl(self) -> FloatMatrix | None:
        """
        ** CACHED PROPERTY **

        2d array of shape (n_pl, n_pix) containing power law flux density
        contributions for each sample.

        Returns
        -------
        y_pl: FloatMatrix | None
            2d array of shape (n_pl, n_pix). If only the base sample has a power
            law model, 'n_pl'=1. Otherwise, 'n_pl'='n_samples'. If the base
            sample has no power law model, None is returned.
        """
        if self.error_result[0].pl is None:
            return None
        elif self.n_samples > 1 and self.error_result[1].pl is None:
            return self.error_result[0].pl(self.x)[None, :]
        return stack([s.pl(self.x) for s in self.error_result], axis=0)

    @property
    def n_pl(self) -> int:
        """
        Number of power law model contributions.
        """
        return 0 if self.y_pl is None else self.y_pl.shape[0]

    @cached_property
    def y_fe(self) -> FloatMatrix | None:
        """
        ** CACHED PROPERTY **

        2d array of shape (n_fe, n_pix) containing Iron continuum flux density
        contributions for each sample.

        Returns
        -------
        y_fe: FloatMatrix | None
            2d array of shape (n_fe, n_pix). If only the base sample has an
            Iron continuum model, 'n_fe'=1. Otherwise, 'n_fe'='n_samples'. If the
            base sample has no Iron continuum model, None is returned.
        """
        if self.error_result[0].fe is None:
            return None
        elif self.n_samples > 1 and self.error_result[1].fe is None:
            return self.error_result[0].fe(self.x)[None, :]
        return stack([s.fe(self.x) for s in self.error_result], axis=0)

    @property
    def n_fe(self) -> int:
        """
        Number of Iron continuum model contributions.
        """
        return 0 if self.y_fe is None else self.y_fe.shape[0]

    @cached_property
    def y_ba(self) -> FloatMatrix | None:
        """
        ** CACHED PROPERTY **

        2d array of shape (n_ba, n_pix) containing Balmer continuum flux density
        contributions for each sample.

        Returns
        -------
        y_ba: FloatMatrix | None
            2d array of shape (n_ba, n_pix). If only the base sample has a Balmer
            continuum model, 'n_ba'=1. Otherwise, 'n_ba'='n_samples'. If the base
            sample has no Balmer continuum model, None is returned.
        """
        if self.error_result[0].ba is None:
            return None
        elif self.n_samples > 1 and self.error_result[1].ba is None:
            return self.error_result[0].ba(self.x)[None, :]
        return stack([s.ba(self.x) for s in self.error_result], axis=0)

    @property
    def n_ba(self) -> int:
        """
        Number of Balmer continuum model contributions.
        """
        return 0 if self.y_ba is None else self.y_ba.shape[0]

    @cached_property
    def y_em(self) -> FloatMatrix | None:
        """
        ** CACHED PROPERTY **

        2d array of shape (n_em, n_pix) containing emission line flux density
        contributions for each sample.

        Returns
        -------
        y_em: FloatMatrix | None
            2d array of shape (n_em, n_pix). If only the base sample has an
            emission line model, 'n_em'=1. Otherwise, 'n_em'='n_samples'. If the
            base sample has no emission line model, None is returned.
        """
        if self.error_result[0].em is None:
            return None
        elif self.n_samples > 1 and self.error_result[1].em is None:
            return self.error_result[0].em(self.x)[None, :]
        return stack([s.em(self.x) for s in self.error_result], axis=0)

    @property
    def n_em(self) -> int:
        """
        Number of emission line model contributions.
        """
        return 0 if self.y_em is None else self.y_em.shape[0]

    @cached_property
    def y_samples(self) -> FloatMatrix:
        """
        ** CACHED PROPERTY **

        2d array of shape (n_samples, n_pix) containing the flux density array
        for each sample.

        Returns
        -------
        y_samples: FloatMatrix
            2d array of shape (n_samples, n_pix). If all samples have identical
            flux density arrays, a 2d array of shape (1, n_pix) containing the
            base sample's flux density array is returned.

        Notes
        -----
        Bootstrapping methods resample the flux density array so the output
        matrix has shape (n_samples, n_pix).

        Nested sampling methods use the sample flux density array so the output
        matrix has shape (1, n_pix). If a non-static background model
        is not subtracted to isolate a specific model component this is not the
        case.
        """
        y_samples = stack([s.y for s in self.error_result], axis=0)
        if self.n_samples > 1 and allclose(y_samples[1:] - y_samples[:1], 0):
            return y_samples[:1]
        return y_samples

    @property
    def n_y_samples(self) -> int:
        """
        Number of resampled flux density arrays.
        """
        return self.y_samples.shape[0]

    @cached_property
    def em_params(self) -> dict[str, list[FloatVector]] | None:
        if self.n_em == 0:
            return None

        samples = list(self.error_result)[: self.n_em]
        result = {
            "mus": [sample.em.mu for sample in samples],
            "sigmas": [sample.em.sigma for sample in samples],
            "strengths": [sample.em.strength for sample in samples],
        }
        result["norm_strengths"] = [s / s.sum() for s in result["strengths"]]

        return result

    @cached_property
    def models_are_stacked(self) -> dict | None:
        """
        ** CACHED PROPERTY **

        Dictionary of stacked Gaussian submodel parameters, i.e. 2d arrays of
        shape (n_samples, ...). If all samples aren't based on the same number
        of Gaussian components, None is returned.

        Notes
        -----
        The boolean value of this property is True, when a dictionary is
        returned, and False, when None is returned.
        """
        if not self.em_params:
            return None

        n = None
        for mu in self.em_params["mus"]:
            if n is not None and mu.size != n:
                return None
            n = mu.size

        return {
            key: stack(value, axis=0) for key, value in self.em_params.items()
        }

    @cached_property
    def fixed_v_profile(self) -> bool:
        """
        ** CACHED PROPERTY **

        Whether all Gaussian submodel parameters indicate a fixed velocity
        profile, i.e. all centres, widths and relative strengths are identical
        across all samples.
        """
        if not self.models_are_stacked:
            return False

        def inner(m: FloatMatrix) -> bool:
            return allclose(m[1:], m[:1])

        return all(
            inner(self.models_are_stacked[key])
            for key in ["mus", "sigmas", "norm_strengths"]
        )

    @cached_property
    def samples_are_weighted(self) -> bool:
        if self.n_samples == 1:
            return False
        return not allclose(
            self.sample_weights[1:] - self.sample_weights[:1],
            0,
        )

    @cached_property
    def discretised_coords(self) -> DiscretisedCoords | None:
        """
        ** CACHED PROPERTY **

        Dataclass containing discretised coordinate arrays:
        - x_bounds: 2d array of lower and upper wavelength bounds.
        - xs: 2d array of discretised wavelength coordinates.
        - dxs: 1d array of discretised wavelength coordinate widths.

        Notes
        -----
        Discretised wavelength coordinates are equally spaced in wavelength
        space, not velocity space like the original spectra, due to Gaussian
        models being defined in wavelength space.
        """
        if not self.em_params:
            return None

        x_bounds = empty((self.n_em, 2), dtype=float64)
        if self.fixed_v_profile:
            mus = self.models_are_stacked["mus"]
            sigmas = self.models_are_stacked["sigmas"]
            x_bounds[:, 0] = (mus[0] - self.render_width * sigmas[0]).min()
            x_bounds[:, 1] = (mus[0] + self.render_width * sigmas[0]).max()
        elif self.models_are_stacked:
            mus = self.models_are_stacked["mus"]
            sigmas = self.models_are_stacked["sigmas"]
            x_bounds[:, 0] = (mus - self.render_width * sigmas).min(axis=1)
            x_bounds[:, 1] = (mus + self.render_width * sigmas).max(axis=1)
        else:
            mus = self.em_params["mus"]
            sigmas = self.em_params["sigmas"]
            for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
                x_bounds[i, 0] = (mu - self.render_width * sigma).min()
                x_bounds[i, 1] = (mu + self.render_width * sigma).max()

        dxs = (x_bounds[:, 1] - x_bounds[:, 0]) / self.res

        if self.fixed_v_profile:
            xs = linspace(
                x_bounds[0, 0] + dxs[0] / 2,
                x_bounds[0, 1] - dxs[0] / 2,
                self.res,
            )[None, :]
        else:
            xs = linspace(
                x_bounds[:, 0] + dxs / 2,
                x_bounds[:, 1] - dxs / 2,
                self.res,
            ).T

        return DiscretisedCoords(x_bounds, xs, dxs)

    @validated_lru_cache
    def getMask(
        self,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
        from_model: bool = True,
        simplify: bool = False,
    ) -> BoolMatrix:
        """
        ** CACHED METHOD **

        Returns a boolean mask of shape (n_samples, n_pix) indicating which
        pixels fall within the wavelength bounds for each sample.
        """
        match centre:
            case "centroid":
                cen = self.measureFirstMoment.__unvalidated__(
                    self,
                    from_model=from_model,
                ).val
            case "median":
                cen = self.measureMedian.__unvalidated__(
                    self,
                    from_model=from_model,
                ).val
            case "mode":
                cen = self.measureMode.__wrapped__(
                    self,
                    from_model=from_model,
                ).val
            case "wave":
                cen = full(self.n_samples, self.wave)

        lb = cen.copy()
        ub = cen.copy()

        if (self.v_int is not None) and (self.v_int > 0):
            lb *= 1 - self.v_int
            ub *= 1 + self.v_int
        elif (self.ipv_int is not None) and (self.ipv_int > 0):
            _x_bounds = self._measureIPBounds.__unvalidated__(
                self,
                p=self.ipv_int,
                from_model=from_model,
            )
            x_bounds = stack([_x_bounds[0].val, _x_bounds[1].val], axis=1)
            if self.n_em != self.n_samples:
                lb[:] = x_bounds[0, 0]
                ub[:] = x_bounds[0, 1]
            else:
                lb[:] = x_bounds[:, 0]
                ub[:] = x_bounds[:, 1]
        elif (self.dx_int is not None) and (self.dx_int > 0):
            lb -= self.dx_int
            ub += self.dx_int
        else:
            lb[:] = -inf
            ub[:] = inf

        mask = (lb[:, None] <= self.x[None, :]) & (
            self.x[None, :] < ub[:, None]
        )

        if simplify and mask.shape[0] > 1 and (mask[1:] == mask[:1]).all():
            return mask[:1]
        return mask

    @validated_lru_cache
    def getY(
        self,
        bg_flux: BackgroundFlux | None = None,
        simplify: bool = False,
        ensure_not_negative: bool = False,
    ) -> FloatMatrix:
        """
        ** CACHED METHOD **

        Returns the flux values for all samples, optionally subtracting
        background flux components.

        If 'simplify' is True and all rows of the output matrix are identical, a
        cropped 2d array of shape (1, n_pix) is returned.
        """
        y = self.y_samples.copy()

        if bg_flux is not None:
            if "pl" in bg_flux and (self.n_pl > 0):
                y = y - self.y_pl
            if "fe" in bg_flux and (self.n_fe > 0):
                y = y - self.y_fe
            if "ba" in bg_flux and (self.n_ba > 0):
                y = y - self.y_ba
            if "em" in bg_flux and (self.n_em > 0):
                y = y - self.y_em
            if "hg" in bg_flux:
                pass

        if ensure_not_negative:
            y[y < 0] = 0

        if simplify and y.shape[0] > 1 and allclose(y[1:] - y[:1], 0):
            return y[:1]
        return y

    @validated_lru_cache
    def getYWeights(
        self,
        *,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
        bin_size_weighted: bool = False,
        normalise: bool = False,
        ensure_not_negative: bool = False,
        simplify: bool = False,
    ) -> FloatMatrix:
        """
        ** CACHED METHOD **

        Returns the spectral weights for all samples, optionally subtracting
        background flux components and applying a mask to isolate the target
        line.

        If 'simplify' is True and all rows of the output matrix are identical, a
        cropped 2d array of shape (1, n_pix) is returned.
        """
        Y = self.getY.__unvalidated__(
            self,
            bg_flux=bg_flux,
            ensure_not_negative=ensure_not_negative,
        ).copy()
        mask = self.getMask.__unvalidated__(
            self,
            centre=centre,
            from_model=from_model,
        )[: Y.shape[0]]
        Y[invert(mask)] = 0

        if bin_size_weighted:
            Y /= self.dx[None, :]
        if normalise:
            Y /= Y.sum(axis=1)[:, None]

        if simplify and Y.shape[0] > 1 and allclose(Y[1:] - Y[:1], 0):
            return Y[:1]
        return Y

    @validated_lru_cache
    def getNY(
        self,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
        ensure_not_negative: bool = False,
    ) -> int:
        Y = self.getY.__unvalidated__(
            self,
            bg_flux=bg_flux,
            ensure_not_negative=ensure_not_negative,
            simplify=True,
        )
        mask = self.getMask.__unvalidated__(
            self,
            centre=centre,
            from_model=from_model,
            simplify=True,
        )
        return max(Y.shape[0], mask.shape[0])

    def getOut(self) -> MeasureResult:
        return MeasureResult.create_empty(self.n_samples)

    @validated_lru_cache
    def getMomentParams(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
    ) -> MomentParams:
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})
        return MomentParams(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
            centre=centre,
        )

    @validated_lru_cache
    def measureFirstMoment(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the first moments for all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        if from_model:
            if not self.em_params:
                return out

            n_em = 1 if self.fixed_v_profile else self.n_em
            d = self.models_are_stacked or self.em_params

            func = dot
            mus = d["mus"][:n_em]
            norm_strengths = d["norm_strengths"][:n_em]
            args_iter = zip(norm_strengths, mus)

            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_em),
            )
        else:
            moment_params = self.getMomentParams(
                from_model=from_model,
                bg_flux=bg_flux,
                centre="wave",
            )
            mask = moment_params.mask
            ys = where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = where(mask, moment_params.dys, 0)

            _out = spectrum_inner.nth_moment(
                1,
                moment_params.xs,
                ys,
                moment_params.dxs,
                dys,
                tot=moment_params.tot,
            )

            out.update(_out, index=slice(None, ys.shape[0]))
            out.remove_negatives()

        return out

    @validated_lru_cache
    def measureSecondCentralMoment(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the second moments for all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        moment_params = self.getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )
        m1 = moment_params.m1

        if from_model:
            if not self.em_params:
                return out

            n_em = 1 if self.fixed_v_profile else self.n_em
            d = self.models_are_stacked or self.em_params

            func = inner.weighted_second_moment
            weights = d["norm_strengths"][:n_em]
            first_moments = m1.val[:n_em]
            mus = d["mus"][:n_em]
            sigmas = d["sigmas"][:n_em]
            args_iter = zip(weights, first_moments, mus, sigmas)

            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_em),
            )
        else:
            mask = moment_params.mask
            ys = where(mask, moment_params.ys, 0)
            if (ys.shape[0] > 1) and allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = where(mask, moment_params.dys, 0)
            _out = spectrum_inner.nth_central_moment(
                2,
                moment_params.xs,
                ys,
                moment_params.dxs,
                dys,
                m1=m1,
                tot=moment_params.tot,
            )

            out.update(_out, index=slice(None, ys.shape[0]))
            out.remove_negatives()

        return out

    @validated_lru_cache
    def measureThirdCentralMoment(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the third moments for all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        moment_params = self.getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )

        if from_model:
            if not self.em_params:
                return out

            n_em = 1 if self.fixed_v_profile else self.n_em
            d = self.models_are_stacked or self.em_params

            func = inner.weighted_third_moment
            args_iter = zip(
                d["norm_strengths"][:n_em],
                moment_params.m1.val[:n_em],
                moment_params.cm2.val[:n_em],
                d["mus"][:n_em],
                d["sigmas"][:n_em],
            )
            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_em),
            )
        else:
            mask = moment_params.mask
            ys = where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = where(mask, moment_params.dys, 0)
            _out = spectrum_inner.nth_central_moment(
                3,
                moment_params.xs,
                ys,
                moment_params.dxs,
                dys,
                m1=moment_params.m1,
                tot=moment_params.tot,
            )

            out.update(_out, index=slice(None, ys.shape[0]))

        return out

    @validated_lru_cache
    def measureFourthCentralMoment(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the fourth moments for all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        moment_params = self.getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )

        if from_model:
            if not self.em_params:
                return out

            n_em = 1 if self.fixed_v_profile else self.n_em
            d = self.models_are_stacked or self.em_params

            func = inner.weighted_fourth_moment
            args_iter = zip(
                d["norm_strengths"][:n_em],
                moment_params.m1.val[:n_em],
                moment_params.cm3.val[:n_em],
                d["mus"][:n_em],
                d["sigmas"][:n_em],
            )
            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_em),
            )
        else:
            mask = moment_params.mask
            ys = where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = where(mask, moment_params.dys, 0)
            _out = spectrum_inner.nth_central_moment(
                4,
                moment_params.xs,
                ys,
                moment_params.dxs,
                dys,
                m1=moment_params.m1,
                tot=moment_params.tot,
            )

            out.update(_out, index=slice(None, ys.shape[0]))
            out.remove_negatives()

        return out

    @validate_call
    def measureCentroid(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Equivalent to the first moment.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        return self.measureFirstMoment.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureMeanOffset(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the offset of the centroid (first moment) from the
        theoretical rest wavelength.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        m1 = self.measureFirstMoment.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return m1 / self.wave - 1

    @validate_call
    def measureDispersion(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the velocity dispersion of all samples.

        Notes
        -----
        This method uses the centroid (first moment) to convert from wavelength
        dispersion to velocity dispersion.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})
        moment_params = self.getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )
        if from_model:
            out = moment_params.cm2**0.5 / moment_params.m1
        else:
            out = self.getOut()

            mask = moment_params.mask
            ys = where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = where(mask, moment_params.dys, 0)
            _out = spectrum_inner.dispersion(
                moment_params.xs,
                ys,
                moment_params.dxs,
                dys,
                tot=moment_params.tot,
                m1=moment_params.m1,
                cm2=moment_params.cm2,
            )

            print(
                moment_params.xs.shape,
                ys.shape,
                moment_params.dxs.shape,
                dys.shape,
            )
            out.update(_out, index=slice(None, ys.shape[0]))
            out.remove_negatives()

        return out

    @validate_call
    def measureSkewness(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the skewness of all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        moment_params = self.getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )
        if from_model:
            out = moment_params.cm3 / moment_params.cm2 ** (3 / 2)
        else:
            out = self.getOut()

            mask = moment_params.mask
            ys = where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = where(mask, moment_params.dys, 0)
            _out = spectrum_inner.skewness(
                moment_params.xs,
                ys,
                moment_params.dxs,
                dys,
                tot=moment_params.tot,
                m1=moment_params.m1,
                cm2=moment_params.cm2,
                cm3=moment_params.cm3,
            )
            out.update(_out, index=slice(None, ys.shape[0]))

        return out

    @validated_lru_cache
    def _measureKurtosis(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the kurtosis of all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        moment_params = self.getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )
        if from_model:
            out = moment_params.cm4 / moment_params.cm2**2
        else:
            out = self.getOut()

            mask = moment_params.mask
            ys = where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = where(mask, moment_params.dys, 0)
            _out = spectrum_inner.kurtosis(
                moment_params.xs,
                ys,
                moment_params.dxs,
                dys,
                tot=moment_params.tot,
                m1=moment_params.m1,
                cm2=moment_params.cm2,
                cm3=moment_params.cm3,
                cm4=moment_params.cm4,
            )
            out.update(_out, index=slice(None, ys.shape[0]))

        return out

    @validated_lru_cache
    def measureKurtosis(
        self,
        from_model: bool = True,
        excess: bool = False,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the kurtosis of all samples. If 'excess' is set to True, the
        excess kurtosis is calculated.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        kurt = self._measureKurtosis.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return kurt - (3.0 if excess else 0.0)

    @validated_lru_cache
    def measureStrength(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the total integrated line strengths.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        if from_model:
            # ? No check for being a vel.-profile copy as the strength can still
            # ? vary in this case.
            n_em = self.n_em
            d = self.models_are_stacked or self.em_params
            out.update_with_single_values(
                [s.sum() for s in d["strengths"][:n_em]],
                index=slice(None, n_em),
            )
        else:
            moment_params = self.getMomentParams(
                from_model=False,
                bg_flux=bg_flux,
                centre="wave",
            )
            mask = moment_params.mask
            ys = where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = where(mask, moment_params.dys, 0)
            out.update(
                spectrum_inner.total(ys, moment_params.dxs, dys),
                index=slice(None, ys.shape[0]),
            )

        return out

    @validated_lru_cache
    def measureQuantile(
        self,
        q: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the wavelength value corresponding to the given quantile for
        all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        if from_model:
            if self.fixed_v_profile:
                n_em = 1
            elif self.models_are_stacked or self.em_params:
                n_em = self.n_em
            else:
                return out

            qs = repeat(q, n_em)
            xs = self.discretised_coords.xs[:n_em]
            cdfs = (sample.em.cdf for sample in self.error_result)
            if self.exact:
                func = exact.get_quantile
                args_iter = zip(qs, cdfs, xs)
            else:
                func = approx.get_quantile
                ys = (cdf(x) for cdf, x in zip(cdfs, xs))
                dxs = self.discretised_coords.dxs[:n_em]
                args_iter = zip(qs, xs, ys, dxs)

            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_em),
            )
        else:
            # Check if simplification is possible
            n_y = self.getNY.__unvalidated__(
                self,
                from_model=True,
                bg_flux=bg_flux,
            )
            Y = self.getY.__unvalidated__(self, bg_flux=bg_flux)[:n_y]
            mask = self.getMask.__unvalidated__(self, from_model=True)[:n_y]

            func = approx.get_quantile
            qs = repeat(q, n_y)
            xs = (self.x[m] for m in mask)
            ys = (y[m] for y, m in zip(Y, mask))
            dxs = (self.dx[m] for m in mask)
            args_iter = zip(qs, xs, ys, dxs)

            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_y),
            )

        return out

    @validated_lru_cache
    def measureMedian(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the wavelength value corresponding to the median for all
        samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        return self.measureQuantile.__unvalidated__(
            self,
            q=0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validated_lru_cache
    def measureMedianOffset(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the velocity-offset of the median w.r.t. the theoretical
        rest-frame wavelength for all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        m = self.measureMedian.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return m / self.wave - 1

    @validated_lru_cache
    def _measureIPBounds(
        self,
        p: float = 10,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> tuple[MeasureResult, MeasureResult]:
        """
        Calculates the inter-percentile wavelength bounds for all samples.

        Notes
        -----
        The inter-percentile bounds for 'p=10' encompass 90% of the total flux.
        The output shape is (n_samples, 2).
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        kwargs = {"from_model": from_model, "bg_flux": bg_flux}
        return stack(
            [
                self.measureQuantile.__unvalidated__(self, p / 200, **kwargs),
                self.measureQuantile.__unvalidated__(
                    self, 1 - p / 200, **kwargs
                ),
            ],
            axis=1,
        )

    @validate_call
    def measureIPWidth(
        self,
        p: float = 10,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the inter-percentile wavelength width for all samples, i.e.
        the distanced between bounds.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        ip_bounds = self._measureIPBounds.__unvalidated__(
            self,
            p=p,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return ip_bounds[1] - ip_bounds[0]

    @validated_lru_cache
    def measureIPV(
        self,
        p: float = 10,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the inter-percentile velocity width for all samples.

        Notes
        -----
        This method uses the median to convert from the wavelength width to
        velocity width.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        ip_width = self.measureIPWidth.__unvalidated__(
            self,
            p=p,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        m = self.measureMedian.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return ip_width / m

    @validated_lru_cache
    def _measureWhittleBounds(
        self,
        p: float = 80,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> tuple[MeasureResult, MeasureResult]:
        """
        Calculates the Whittle bounds, a and b, useful for calculating other
        Whittle parameters.

        Notes
        -----
        The shape of an output numpy array is: (n_samples, 2)
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        ip_bounds = self._measureIPBounds.__unvalidated__(
            self,
            p=p,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        m = self.measureMedian.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (
            abs(ip_bounds[0] / m - 1),
            abs(ip_bounds[1] / m - 1),
        )

    @validate_call
    def measureWhittleA(
        self,
        p: float = 10,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the Whittle 'a' bound.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        return self._measureWhittleBounds.__unvalidated__(
            self,
            p=p,
            from_model=from_model,
            bg_flux=bg_flux,
        )[0].copy()

    @validate_call
    def measureWhittleB(
        self,
        p: float = 10,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the Whittle 'b' bound.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        return self._measureWhittleBounds.__unvalidated__(
            self,
            p=p,
            from_model=from_model,
            bg_flux=bg_flux,
        )[1].copy()

    @validate_call
    def measureWhittleShift(
        self,
        p: float = 10,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the Whittle shift parameter for each sample: S = a - b.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        a, b = self._measureWhittleBounds.__unvalidated__(
            self,
            p=p,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return a - b

    @validate_call
    def measureWhittleAsymmetry(
        self,
        p: float = 10,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the Whittle asymmetry parameter for each sample:
            A = (a - b) / (a + b)
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        a, b = self._measureWhittleBounds.__unvalidated__(
            self,
            p=p,
            from_model=from_model,
            bg_flux=bg_flux,
        )

        mask = (a.val + b.val) > 0
        _num = a.val[mask] - b.val[mask]
        _den = a.val[mask] + b.val[mask]

        out = self.getOut()
        out.val[mask] = _val = _num / _den
        out.unc[mask] = (
            1
            / _den
            * hypot(
                (1 - _val) * a.unc[mask],
                (1 + _val) * b.unc[mask],
            )
        )
        return out

    @validate_call
    def measureWhittleK1(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the Whittle K1 parameter for each sample:
            K1 = 1.524 * (FWHM / FW20)
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        fwhm = self.measureFWHM.__unvalidated__(
            self,
            h=0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        fw20 = self.measureFWHM.__unvalidated__(
            self,
            h=0.2,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return 1.524 * fwhm / fw20

    @validate_call
    def measureWhittleK2(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the Whittle K2 parameter for each sample:
            K2 = 1.397 * (FWHM / IPW(10%))
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        fwhms = self.measureFWHM.__unvalidated__(
            self,
            h=0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        ipw10 = self.measureIPWidth.__unvalidated__(
            self,
            p=10,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return 1.397 * fwhms / ipw10

    @validated_lru_cache
    def _measurePeak(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> tuple[MeasureResult, MeasureResult]:
        """
        Calculates the peak flux density value and its location for each sample.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        modes = self.getOut()
        maxs = self.getOut()

        if from_model:
            if self.fixed_v_profile:
                n_em = 1
            elif self.models_are_stacked or self.em_params:
                n_em = self.n_em
            else:
                return (modes, maxs)

            ems = (sample.em for sample in self.error_result)
            xs = self.discretised_coords.xs[:n_em]

            if self.exact:
                func = exact.get_peak
                args_iter = zip(ems, xs)
            else:
                func = approx.get_peak
                ys = (em(x) for em, x in zip(ems, xs))
                dxs = self.discretised_coords.dxs[:n_em]
                args_iter = zip(xs, ys, dxs)

            results = list(self.starmap(func, args_iter))
        else:
            n_y = self.getNY.__unvalidated__(
                self,
                from_model=True,
                bg_flux=bg_flux,
            )
            Y = self.getY.__unvalidated__(self, bg_flux=bg_flux)[:n_y]
            mask = self.getMask.__unvalidated__(self, from_model=True)[:n_y]

            func = approx.get_peak
            xs = (self.x[m] for m in mask)
            ys = (y[m] for y, m in zip(Y, mask))
            dxs = (self.dx[m] for m in mask)
            args_iter = zip(xs, ys, dxs)

            results = list(self.starmap(func, args_iter))

        modes.update_with_single_values(
            [r[0] for r in results],
            index=slice(None, len(results)),
        )
        maxs.update_with_single_values(
            [r[1] for r in results],
            index=slice(None, len(results)),
        )
        return (modes, maxs)

    @validate_call
    def measureMode(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the mode for all samples, i.e. the wavelength-position of the
        peak.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        return self._measurePeak.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )[0]

    @validate_call
    def measureModeOffset(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the velocity offset of the mode for all samples w.r.t. the
        theoretical rest-frame wavelength.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        mode = self.measureMode.__wrapped__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return mode / self.wave - 1

    @validate_call
    def measureMax(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the peak flux density value for all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        return self._measurePeak.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )[1].copy()

    @validated_lru_cache
    def measureFWHM(
        self,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the full wavelength width of the profile at a given ratio
        of the peak flux density.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        peak = self._measurePeak.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        if from_model:
            if self.fixed_v_profile:
                n_em = 1
            elif self.models_are_stacked or self.em_params:
                n_em = self.n_em
            else:
                return out

            models = (sample.em for sample in self.error_result)
            xs = self.discretised_coords.xs[:n_em]
            x_maxs = peak[0].val[:n_em]
            y_crits = h * peak[1].val[:n_em]

            if self.exact:
                args_iter = zip(models, xs, x_maxs, y_crits)

                def func(model, x, x_max, y_crit):
                    return exact.get_width_at_height(
                        model,
                        x,
                        x_max,
                        y_crit,
                        self.fwhm_strategy,
                    )
            else:
                dy = ones(self.res, dtype=float64)
                ys = (model(x) for model, x in zip(models, xs))
                args_iter = zip(xs, ys, x_maxs, y_crits)

                def func(x, y, x_max, y_crit):
                    return approx.get_width_at_height(
                        x,
                        y,
                        dy,
                        x_max,
                        y_crit,
                        self.fwhm_strategy,
                    )[0]

            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_em),
            )
        else:
            moment_params = self.getMomentParams.__unvalidated__(
                self,
                from_model=False,
                bg_flux=bg_flux,
                centre="mode",
            )
            n_y = self.getNY.__unvalidated__(
                self,
                centre="mode",
                from_model=False,
                bg_flux=bg_flux,
            )
            mask = moment_params.mask[:n_y]

            xs = (self.x[m] for m in mask)
            ys = (y[m] for y, m in zip(moment_params.ys, mask))
            dys = (self.dy[m] for m in mask)
            x_maxs = peak[0].val[:n_y]
            y_crits = h * peak[1].val[:n_y]
            args_iter = zip(xs, ys, dys, x_maxs, y_crits)

            # return list(xs), list(ys), list(dys), list(x_maxs), list(y_crits)

            def func(x, y, dy, x_max, y_crit):
                return approx.get_width_at_height(
                    x,
                    y,
                    dy,
                    x_max,
                    y_crit,
                    self.fwhm_strategy,
                )

            out.update_with_list_of_tuples(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_y),
            )

        return out

    @validate_call
    def measureFWHMVelocity(
        self,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the FWHM for all samples w.r.t. their modes, i.e. the
        rest wavelengths of their peak flux densities.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        mode = self.measureMode.__wrapped__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        fwhm = self.measureFWHM.__unvalidated__(
            self,
            h,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return fwhm / mode

    @validated_lru_cache
    def measureEW(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Notes
        -----
        No analytical solution, so a numerical integral is used instead.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        if from_model:
            n_em = self.n_em
            n_pl = self.n_pl
            n = max(n_em, n_pl)

            xs = self.discretised_coords.xs
            dxs = self.discretised_coords.dxs

            weights = empty((n, self.res), dtype=float64)
            if n_em == n:
                weights[:] = [
                    sample.em(x) * dx
                    for sample, x, dx in zip(self.error_result, xs, dxs)
                ]
            else:
                weights[:] = self.error_result[0].em(xs[:1]) * dxs[:1]

            inv_continua = empty((n, self.res), dtype=float64)
            if n_pl == n_em:
                inv_continua[:] = [
                    1 / sample.pl(x)
                    for sample, x in zip(self.error_result, xs)
                ]
            elif n_pl == n:
                inv_continua[:] = [
                    1 / sample.pl(xs[0]) for sample in self.error_result
                ]
            else:
                inv_continua[:] = 1 / self.error_result[0].pl(xs[:1])

            out.update_with_single_values(
                einsum("ij,ij->i", weights, inv_continua),
                index=slice(None, n),
            )
        else:
            n_y = self.getNY.__unvalidated__(
                self,
                from_model=True,
                bg_flux=bg_flux,
            )
            Y = self.getY.__unvalidated__(
                self,
                bg_flux=bg_flux,
            )[:n_y]
            mask = self.getMask.__unvalidated__(
                self,
                from_model=True,
            )[:n_y]

            n = max(n_y, self.n_pl)
            weights = empty((n, self.n_pix), dtype=float64)
            if n_y == n:
                weights[:] = where(mask, Y, 0) * self.dx[None, :]
            else:
                weights[:] = (where(mask[0], Y[0], 0) * self.dx)[None, :]

            inv_continua = empty((n, self.n_pix), dtype=float64)
            if self.n_pl == n:
                inv_continua[:] = 1 / self.y_pl
            else:
                inv_continua[:] = 1 / self.y_pl[:1]

            out.update_with_single_values(
                einsum("ij,ij->i", weights, inv_continua),
                index=slice(None, n),
            )

        return out

    @validated_lru_cache
    def measureMAD(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the median absolute deviation (MAD) in units of wavelength
        for all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        median = self.measureMedian.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        if from_model:
            ems = (sample.em for sample in self.error_result)
            n_em = 1 if self.fixed_v_profile else self.n_em

            cens = median[:n_em]
            xs = self.discretised_coords.xs[:n_em]
            if self.exact:
                func = exact.get_median_absolute_deviation
                cdfs = (em.cdf for em in ems)
                args_iter = zip(cdfs, cens, xs)
            else:
                func = approx.get_median_absolute_deviation
                ys = (em(x) for em, x in zip(ems, xs))
                dxs = self.discretised_coords.dxs[:n_em]
                args_iter = zip(cens, xs, ys, dxs)

            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_em),
            )
        else:
            n_y = self.getNY.__unvalidated__(
                self,
                centre="median",
                from_model=True,
                bg_flux=bg_flux,
            )
            Y = self.getY.__unvalidated__(
                self,
                bg_flux=bg_flux,
            )[:n_y]
            mask = self.getMask.__unvalidated__(
                self,
                centre="median",
                from_model=True,
            )[:n_y]

            func = approx.get_median_absolute_deviation
            cens = median[:n_y]
            xs = (self.x[m] for m in mask)
            ys = (y[m] for y, m in zip(Y, mask))
            dxs = (self.dx[m] for m in mask)
            args_iter = zip(cens, xs, ys, dxs)

            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_y),
            )

        return out

    @validate_call
    def measureMADVelocity(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the median absolute deviation (MAD) in units of velocity
        w.r.t. the median for all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        med = self.measureMedian.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        mad = self.measureMAD.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return mad / med

    @validated_lru_cache
    def measureAAD(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the average absolute deviation in units of wavelength for all
        samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut()
        centroid = self.measureCentroid.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        if from_model:
            n_em = 1 if self.fixed_v_profile else self.n_em

            cens = centroid.val[:n_em]
            if self.exact:
                func = exact.get_average_absolute_deviation
                weights = (sample.em.weight for sample in self.error_result)
                dists = (sample.em.dist for sample in self.error_result)
                args_iter = zip(cens, dists, weights)
            else:
                func = approx.get_average_absolute_deviation
                xs = self.discretised_coords.xs[:n_em]
                ys = (sample.em(x) for sample, x in zip(self.error_result, xs))
                dxs = self.discretised_coords.dxs[:n_em]
                args_iter = zip(cens, xs, ys, dxs)

            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_em),
            )
        else:
            n_y = self.getNY.__unvalidated__(
                self,
                centre="centroid",
                from_model=True,
                bg_flux=bg_flux,
            )
            Y = self.getY.__unvalidated__(
                self,
                bg_flux=bg_flux,
            )[:n_y]
            mask = self.getMask.__unvalidated__(
                self,
                centre="centroid",
                from_model=True,
            )[:n_y]

            func = approx.get_average_absolute_deviation
            cens = centroid.val[:n_y]
            xs = (self.x[m] for m in mask)
            ys = (y[m] for y, m in zip(Y, mask))
            dxs = (self.dx[m] for m in mask)
            args_iter = zip(cens, xs, ys, dxs)

            out.update_with_single_values(
                list(self.starmap(func, args_iter)),
                index=slice(None, n_y),
            )

        return out

    @validate_call
    def measureAADVelocity(
        self,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the average absolute deviation in units of velocity w.r.t.
        the centroid for all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        centroid = self.measureFirstMoment.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        aad = self.measureAAD.__unvalidated__(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return aad / centroid

    @validated_lru_cache
    def measureIntegratedFlux(
        self,
        x_bounds: tuple[float, float],
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the integrated flux across a given wavelength range for
        all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux(set())

        out = self.getOut()
        mask = (x_bounds[0] <= self.x) & (self.x <= x_bounds[1])
        if not mask.any():
            return out

        dxs = self.dx[mask][None, :]

        if from_model:
            ys = 0
            n = 0
            if ("pl" not in bg_flux) and self.n_pl > 0:
                n = max(n, self.n_pl)
                ys += self.y_pl[:, mask]
            if ("fe" not in bg_flux) and self.n_fe > 0:
                n = max(n, self.n_fe)
                ys += self.y_fe[:, mask]
            if ("ba" not in bg_flux) and self.n_ba > 0:
                n = max(n, self.n_ba)
                ys += self.y_ba[:, mask]
            if ("em" not in bg_flux) and self.n_em > 0:
                n = max(n, self.n_em)
                ys += self.y_em[:, mask]

            out.update_with_single_values(
                (ys * dxs).sum(axis=1),
                index=slice(None, n),
            )
        else:
            ys = self.getY.__unvalidated__(
                self,
                bg_flux=bg_flux,
                simplify=True,
            )[:, mask]
            dys = self.dy[mask][None, :]

            out.update(
                spectrum_inner.total(ys, dxs, dys),
                index=slice(None, ys.shape[0]),
            )

        return out

    @validated_lru_cache
    def measureAverageFluxDensity(
        self,
        x_bounds: tuple[float, float],
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        if bg_flux is None:
            bg_flux = BackgroundFlux(set())

        out = self.getOut()
        mask = (x_bounds[0] <= self.x) & (self.x <= x_bounds[1])
        if not mask.any():
            return out

        integrated_flux = self.measureIntegratedFlux.__unvalidated__(
            self,
            x_bounds=x_bounds,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return integrated_flux / self.dx[mask].sum()

    @validated_lru_cache
    def measurePLFlux(
        self,
        x: float = 1450,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the power law continuum flux density at a given point for all
        samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "pl"})

        out = self.getOut()
        if from_model or ((x < self.x[0]) or (self.x[-1] < x)):
            if self.n_pl == 0:
                return out

            out.update_with_single_values(
                [s.pl(x) for _, s in zip(range(self.n_pl), self.error_result)],
                index=slice(None, self.n_pl),
            )

        else:
            ys = self.getY.__unvalidated__(
                self,
                bg_flux=bg_flux,
                simplify=True,
            )
            out.update_with_list_of_tuples(
                [approx.u_yinterp(x, self.x, y, self.dy) for y in ys],
                index=slice(None, ys.shape[0]),
            )

        return out

    @validated_lru_cache
    def measureFEFlux(
        self,
        x: float = 1450,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the iron pseudo-continuum flux density at a given point for
        all samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "fe"})

        out = self.getOut()
        if from_model or ((x < self.x[0]) or (self.x[-1] < x)):
            if self.n_fe == 0:
                return out

            out.update_with_single_values(
                [s.fe(x) for _, s in zip(range(self.n_fe), self.error_result)],
                index=slice(None, self.n_fe),
            )
        else:
            ys = self.getY.__unvalidated__(
                self,
                bg_flux=bg_flux,
                simplify=True,
            )
            out.update_with_list_of_tuples(
                [approx.u_yinterp(x, self.x, y, self.dy) for y in ys],
                index=slice(None, ys.shape[0]),
            )

        return out

    @validated_lru_cache
    def measureBAFlux(
        self,
        x: float = 1450,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> MeasureResult:
        """
        Calculates the Balmer continuum flux density at a given point for all
        samples.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "ba"})

        out = self.getOut()
        if from_model or ((x < self.x[0]) or (self.x[-1] < x)):
            if self.n_ba == 0:
                return out

            out.update_with_single_values(
                [s.ba(x) for _, s in zip(range(self.n_ba), self.error_result)],
                index=slice(None, self.n_ba),
            )
        else:
            ys = self.getY.__unvalidated__(
                self,
                bg_flux=bg_flux,
                simplify=True,
            )
            out.update_with_list_of_tuples(
                [approx.u_yinterp(x, self.x, y, self.dy) for y in ys],
                index=slice(None, ys.shape[0]),
            )

        return out
