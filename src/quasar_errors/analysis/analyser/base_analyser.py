from collections.abc import Callable, Iterable, Iterator
from dataclasses import field
from functools import cached_property, wraps
from itertools import starmap, takewhile
from typing import ClassVar, Literal, Self, TypedDict

import numpy as np
import pydantic
from astropy.units import Unit
from quasar_typing.astropy import CompositeUnitType, UnitType
from quasar_typing.literals import FWHMStrategy
from quasar_typing.multiprocessing import PoolType
from quasar_typing.numpy import BoolMatrix, FloatMatrix, FloatVector
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.decorators import lru_cache
from quasar_utils.setup import Info

from ...error_result import ErrorResult
from ..measure_result import MeasureResult
from ..moment_params import MomentParams


def _can_simplify_matrix(mat: FloatMatrix) -> bool:
    return mat.shape[0] > 1 and all(
        np.array_equal(mat[0,:], arr)
        for arr in mat[1:]
    )

class EMParams(TypedDict):
    mus: list[FloatVector]
    sigmas: list[FloatVector]
    strengths: list[FloatVector]
    norm_strengths: list[FloatVector]

class StackedEMParams(TypedDict):
    mus: FloatMatrix
    sigmas: FloatMatrix
    strengths: FloatMatrix
    norm_strengths: FloatMatrix

class DiscretisedCoords(TypedDict):
    x_bounds: FloatMatrix
    xs: FloatMatrix
    dxs: FloatVector


@pydantic.dataclasses.dataclass(repr=False)
class _BaseAnalyser:
    """
    Notes
    -----
    Private methods are prefixed with an underscore (_) and are intended for 
    internal use within the class, or use by an experienced developer. These 
    methods are sometimes cached (LRU). 

    External methods have no underscore prefix and are intended for the general 
    user. They are simple wrappers of their corresponding private methods using 
    `pydantic.validate_call`.
    """
    error_result: ErrorResult

    pool: PoolType | None = field(default=None, kw_only=True)
    info: Info = field(default_factory=Info, kw_only=True)

    # Hyperparameters
    res: int | None = field(default=None, kw_only=True)
    render_width: float | None = field(default=None, kw_only=True)
    fwhm_strategy: FWHMStrategy | None = field(default=None, kw_only=True)
    v_int: float | None = field(default=None, kw_only=True)
    ipv_int: float | None = field(default=None, kw_only=True)
    dx_int: float | None = field(default=None, kw_only=True)

    line: str | set[str] | None = field(init=False)
    _waves: set[float] | None = field(init=False, repr=False)
    wave: float | None = field(init=False, repr=False)

    sample_weights: FloatVector = field(init=False)
    x: FloatVector = field(init=False)
    y: FloatVector = field(init=False)
    dx: FloatVector = field(init=False)
    dy: FloatVector = field(init=False)
    n_samples: int = field(init=False)
    n_pix: int = field(init=False)

    # Correction for when IPV bounds are used. 
    mask_correction: float = field(default=1.0, init=False)

    methods: ClassVar[frozenset[str]]
    measures: ClassVar[frozenset[str]]

    @classmethod
    @pydantic.validate_call
    def create(
        cls, 
        error_result: ErrorResult,
        *,
        target_pattern: str | tuple[str, ...] | Callable[[str], bool] | None = None,
        target_wave: float | set[float] | None = None,
        target_line: str | set[str] | None = None,
        pool: PoolType | None = None,
        info: Info | None = None,
        res: int | None = None,
        render_width: float | None = None,
        fwhm_strategy: FWHMStrategy | None = None,
        v_int: float | None = None,
        ipv_int: float | None = None,
        dx_int: float | None = None,
    ) -> Self:
        er = error_result.isolate.__wrapped__(
            error_result,
            target_pattern=target_pattern,
            target_wave=target_wave,
            target_line=target_line,
        )
        return cls(
            er,
            pool=pool,
            info=info or Info(),
            res=res,
            render_width=render_width,
            fwhm_strategy=fwhm_strategy,
            v_int=v_int,
            ipv_int=ipv_int,
            dx_int=dx_int,
        )

    def __post_init__(self) -> None:
        """
        Runs the following:
        - `_set_lines_and_waves`
        - `_set_hyperparams`
        - `_set_data`
        """
        self._set_lines_and_waves()
        self._set_hyperparams()
        self._set_data()

    def _set_lines_and_waves(self) -> None:
        if self.error_result[0].em is None:
            self.line = None
            self._waves = None
            self.wave = None
        else:
            lines = set()
            waves = set()
            for sample in self.error_result[0].em.samples:
                lines.add(sample.pure_name)
                waves.add(sample.wave)

            self.line = lines.pop() if len(lines) == 1 else lines
            self._waves = waves
            self.wave = sum(waves) / len(waves)

    def _set_hyperparams(self) -> None:
        if self.res is None:
            self.res = self.info.error.res
        if self.render_width is None:
            self.render_width = self.info.error.render_width
        if self.fwhm_strategy is None:
            self.fwhm_strategy = self.info.error.fwhm_strategy
        if self.v_int is None:
            self.v_int = self.info.error.v_int
        if self.ipv_int is None:
            self.ipv_int = self.info.error.ipv_int
        if self.dx_int is None:
            self.dx_int = self.info.error.dx_int

    def _set_data(self) -> None:
        self.sample_weights = np.array(
            self.error_result.sample_weights, 
            dtype=np.float64,
        )
        self.x = self.error_result[0].x
        self.y = self.error_result[0].y
        self.dx = self.error_result[0].dx
        self.dy = self.error_result[0].dy
        self.n_samples = self.sample_weights.size
        self.n_pix = self.x.size

    def __repr__(self) -> str:
        line = self.line
        n_samples = self.n_samples
        return f"<{self.__class__.__name__} at {hex(id(self))}> "\
            f"for {line=} w/ {n_samples=}"

    def __tuple__(self) -> tuple:
        """Create a tuple of the current Analyser instance."""
        return (
            id(self.error_result), # ErrorResult instance is always frozen
            self.res, 
            self.render_width,
            self.fwhm_strategy, 
            self.v_int,
            self.ipv_int,
            self.dx_int,
            tuple(self.line) if isinstance(self.line, Iterable) else self.line,
            tuple(self._waves) if isinstance(self._waves, Iterable) else self._waves,
            self.wave,
            tuple(self.sample_weights),
            tuple(self.x),
            tuple(self.y),
            tuple(self.dx),
            tuple(self.dy),
            self.n_samples,
            self.n_pix,
            self.mask_correction,
        )

    @property
    def map(self) -> Callable[[Callable, Iterable], Iterator]:
        return map if self.pool is None else self.pool.map

    @property
    def starmap(self) -> Callable[[Callable, Iterable], Iterator]:
        return starmap if self.pool is None else self.pool.starmap

    # Flux density matrices

    @cached_property
    def y_pl(self) -> FloatMatrix | None:
        """
        Returns a 2d array of all powerlaw flux densities for each sample. 
        If no powerlaw model sample is present, returns None. If only the base 
        sample has a powerlaw model sample, returns the corresponding array with 
        shape (1, n_pix).
        """
        if self.error_result[0].pl is None:
            return None
        elif (self.n_samples > 1) and self.error_result[1].pl is None:
            return self.error_result[0].pl(self.x)[None,:]
        return np.array(
            [s.pl(self.x) for s in self.error_result.samples], 
            np.float64
        )

    @cached_property
    def y_fe(self) -> FloatMatrix | None:
        """
        Returns a 2d array of all iron emission flux densities for each sample. 
        If no iron emission model sample is present, returns None. If only the base 
        sample has an iron emission model sample, returns the corresponding array with 
        shape (1, n_pix).
        """
        if self.error_result[0].fe is None:
            return None
        elif (self.n_samples > 1) and self.error_result[1].fe is None:
            return self.error_result[0].fe(self.x)[None,:]
        return np.array(
            [s.fe(self.x) for s in self.error_result.samples], 
            np.float64
        )

    @cached_property
    def y_ba(self) -> FloatMatrix | None:
        """
        Returns a 2d array of all Balmer continuum flux densities for each sample. 
        If no Balmer continuum model sample is present, returns None. If only the base 
        sample has a Balmer continuum model sample, returns the corresponding array with 
        shape (1, n_pix).
        """
        if self.error_result[0].ba is None:
            return None
        elif (self.n_samples > 1) and self.error_result[1].ba is None:
            return self.error_result[0].ba(self.x)[None,:]
        return np.array(
            [s.ba(self.x) for s in self.error_result.samples],
            np.float64,
        )

    @cached_property
    def y_hg(self) -> FloatMatrix | None:
        """
        Returns a 2d array of all host galaxy flux densities for each sample. 
        If no host galaxy model sample is present, returns None. If only the base 
        sample has a host galaxy model sample, returns the corresponding array with 
        shape (1, n_pix).
        """
        if self.error_result[0].hg is None:
            return None
        elif (self.n_samples > 1) and self.error_result[1].hg is None:
            return self.error_result[0].hg(self.x)[None,:]
        return np.array(
            [s.hg(self.x) for s in self.error_result.samples], 
            np.float64
        )

    @cached_property
    def y_em(self) -> FloatMatrix | None:
        """
        Returns a 2d array of all emission line flux densities for each sample. 
        If no emission line model sample is present, returns None. If only the base 
        sample has an emission line model sample, returns the corresponding array with 
        shape (1, n_pix).
        """
        if self.error_result[0].em is None:
            return None
        elif (self.n_samples > 1) and self.error_result[1].em is None:
            return self.error_result[0].em(self.x)[None,:]
        return np.array(
            [s.em(self.x) for s in self.error_result.samples], 
            np.float64
        )

    @cached_property
    def y_samples(self) -> FloatMatrix:
        """
        Returns a 2d array of all flux densities for each sample. 
        """
        if self.n_samples > 1 and all(
            np.array_equal(
                self.error_result[0].y, 
                self.error_result[i].y,
            ) 
            for i in range(1, self.n_samples)
        ):
            return self.error_result[0].y[None,:]
        return np.array([s.y for s in self.error_result.samples], np.float64)

    # Sizes of matrices

    @property
    def n_pl(self) -> int:
        """Number of unique powerlaw model samples."""
        return 0 if self.y_pl is None else self.y_pl.shape[0]

    @property
    def n_fe(self) -> int:
        """Number of unique iron model samples."""
        return 0 if self.y_fe is None else self.y_fe.shape[0]

    @property
    def n_ba(self) -> int:
        """Number of unique Balmer continuum model samples."""
        return 0 if self.y_ba is None else self.y_ba.shape[0]

    @property
    def n_hg(self) -> int:
        """Number of unique host galaxy model samples."""
        return 0 if self.y_hg is None else self.y_hg.shape[0]

    @property
    def n_em(self) -> int:
        """Number of unique emission line model samples."""
        return 0 if self.y_em is None else self.y_em.shape[0]

    @property
    def n_y_samples(self) -> int:
        """Number of unique flux density samples."""
        return self.y_samples.shape[0]

    # Gaussian parameters

    @cached_property
    def em_params(self) -> EMParams | None:
        if self.n_em == 0:
            return None

        mus: list[FloatVector] = []
        sigmas: list[FloatVector] = []
        strengths: list[FloatVector] = []

        for sample in takewhile(
            lambda s: s.em is not None, 
            self.error_result.samples,
        ):
            mus.append(sample.em.mu)
            sigmas.append(sample.em.sigma)
            strengths.append(sample.em.strength)

        return {
            "mus": mus,
            "sigmas": sigmas,
            "strengths": strengths,
            "norm_strengths": [s / s.sum() for s in strengths],
        }

    @cached_property
    def stacked_em_params(self) -> StackedEMParams | None:
        if self.em_params is None:
            return None

        prev_size = None
        for mu in self.em_params["mus"]:
            curr_size = mu.size
            if (prev_size is not None) and curr_size != prev_size:
                return None
            prev_size = curr_size

        return {
            key: np.array(value, np.float64)
            for key, value in self.em_params.items()
        }

    ###

    @cached_property
    def v_profile_is_fixed(self) -> bool:
        if self.stacked_em_params is None:
            return False

        return all(
            _can_simplify_matrix(m)
            for m in self.stacked_em_params.values()
        )

    @cached_property
    def samples_are_weighted(self) -> bool:
        if self.n_samples == 1:
            return False
        return not all(
            np.array_equal(self.sample_weights[0], self.sample_weights[i]) 
            for i in range(1, self.n_samples)
        )

    ### Coordinates

    @cached_property
    def discretised_coords(self) -> DiscretisedCoords | None:
        if self.em_params is None:
            return None

        x_bounds = np.empty((self.n_em, 2), dtype=np.float64)
        if self.v_profile_is_fixed:
            mu = self.stacked_em_params["mus"][0]
            sigma = self.stacked_em_params["sigmas"][0]
            x_bounds[:,0] = (mu - self.render_width * sigma).min()
            x_bounds[:,1] = (mu + self.render_width * sigma).max()
        elif self.stacked_em_params is not None:
            mus = self.stacked_em_params["mus"]
            sigmas = self.stacked_em_params["sigmas"]
            x_bounds[:,0] = (mus - self.render_width * sigmas).min(axis=1)
            x_bounds[:,1] = (mus + self.render_width * sigmas).max(axis=1)
        else:
            mus = self.em_params["mus"]
            sigmas = self.em_params["sigmas"]
            for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
                x_bounds[i,0] = (mu - self.render_width * sigma).min()
                x_bounds[i,1] = (mu + self.render_width * sigma).max()

        dxs = (x_bounds[:,1] - x_bounds[:,0]) / self.res
        if self.v_profile_is_fixed:
            xs = np.linspace(
                x_bounds[0,0] + dxs[0] / 2,
                x_bounds[0,1] - dxs[0] / 2,
                self.res
            )[None,:]
        else:
            xs = np.linspace(
                x_bounds[:,0] + dxs / 2,
                x_bounds[:,1] - dxs / 2,
                self.res
            ).T
            
        return {
            "x_bounds": x_bounds,
            "xs": xs,
            "dxs": dxs
        }

    @lru_cache
    def _getMask(
        self,
        *,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
        from_model: bool = True,
        simplify: bool = False,
    ) -> BoolMatrix:
        """
        Construct a 2d boolean mask for the flux density matrix, where True 
        indicates that the corresponding element is within the bounds. 

        The bounds are determined using:

        1. `self.v_int > 0`: A velocity offset from the specified centre.
        2. `self.ipv_int > 0`: The inter-percentile velocity bounds.
        3. `self.dx_int > 0`: A fixed wavelength offset from the specified 
        centre.

        Otherwise, all elements of the mask are set to True. 

        Parameters
        ----------
        centre : Literal["wave", "centroid", "median", "mode"], optional
            The centre to use for the bounds. Defaults to "wave".
        from_model : bool, optional
            Whether to use the model or data to determine the centre and/or 
            bounds. Defaults to True.
        simplify : bool, optional
            Whether to simplify the mask to a single row if all rows are equal.
            Defaults to False.

        Returns
        -------
        BoolMatrix
            A 2d boolean mask with shape (n_samples, n_pix).
        
        Notes
        -----
        In the case that the inter-percentile bounds are used, the 
        `mask_correction` attribute is updated to reflect that discrete 
        integrals using this mask should be adjusted by a factor of 
        `1 / self.ipv_int`.
        """
        if centre == "wave":
            lb = np.full(self.n_samples, min(self._waves), dtype=np.float64)
            ub = np.full(self.n_samples, max(self._waves), dtype=np.float64)
        else:
            match centre:
                case "centroid":
                    cen = self._measureFirstMoment(from_model=from_model).val
                case "median":
                    cen = self._measureMedian(from_model=from_model).val
                case "mode":
                    cen = self._measureMode(from_model=from_model).val

            lb = cen.copy()
            ub = cen.copy()

        if (self.v_int > 0):
            lb *= 1 - self.v_int
            ub *= 1 + self.v_int
            self.mask_correction = 1.0
        elif self.ipv_int > 0:
            ipv_bounds = self._measureIPBounds(
                self.ipv_int,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
            x_bounds = np.stack([ipv_bounds[0].val, ipv_bounds[1].val], axis=1)
            self.mask_correction = 100 / (100 - self.ipv_int)

            if self.n_em == self.n_samples:
                lb[:] = x_bounds[:,0]
                ub[:] = x_bounds[:,1]
            else:
                lb[:] = x_bounds[0,0]
                ub[:] = x_bounds[0,1]
        elif self.dx_int > 0:
            lb -= self.dx_int
            ub += self.dx_int
            self.mask_correction = 1.0
        else:
            lb[:] = -np.inf
            ub[:] = np.inf
            self.mask_correction = 1.0

        mask = (lb[:,None] <= self.x[None,:]) & (self.x[None,:] <= ub[:,None])
        return mask[:1] if (simplify and _can_simplify_matrix(mask)) else mask
    
    @lru_cache
    def _getFlux(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        ensure_not_negative: bool = False,
        simplify: bool = False,
    ) -> FloatMatrix:
        """
        Create a 2d array of flux densities excluding the specified background 
        fluxes.

        Parameters
        ----------
        bg_flux : BackgroundFlux | None, optional
            The background fluxes to exclude. If None, defaults to excluding all 
            background fluxes. 
        ensure_not_negative : bool, optional
            If True, any negative flux densities are set to zero. Defaults to 
            False.
        simplify : bool, optional
            Whether to simplify the flux density matrix to a single row if all 
            rows are identical. Defaults to False.

        Returns
        -------
        FloatMatrix
            A 2d array of flux densities with shape (n_samples, n_pix).
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux(set())

        y = self.y_samples.copy()
        if "pl" in bg_flux and self.y_pl is not None:
            y -= self.y_pl
        if "fe" in bg_flux and self.y_fe is not None:
            y -= self.y_fe
        if "ba" in bg_flux and self.y_ba is not None:
            y -= self.y_ba
        if "hg" in bg_flux and self.y_hg is not None:
            y -= self.y_hg
        if "em" in bg_flux and self.y_em is not None:
            y -= self.y_em

        if ensure_not_negative:
            y[y < 0] = 0.0
        if simplify and _can_simplify_matrix(y):
            return y[:1]
        return y

    @lru_cache
    def _getFluxWeights(
        self,
        *,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
        weigh_by_bin_size: bool = False,
        ensure_not_negative: bool = False,
        simplify: bool = False,
        normalize: bool = False,
    ) -> FloatMatrix:
        """
        Create a 2d array of weights based on flux density arrays.

        Parameters
        ----------
        centre : Literal["wave", "centroid", "median", "mode"], optional
            See `getMask` for details. Defaults to "wave".
        from_model : bool, optional
            See `getMask` for details. Defaults to True.
        bg_flux : BackgroundFlux | None, optional
            See `getFlux` for details. Defaults to None.
        weigh_by_bin_size : bool, optional
            If True, the weights are multiplied by the bin sizes s.t. weights 
            are based on flux rather than flux density. Defaults to False.
        ensure_not_negative : bool, optional
            See `getFlux` for details. Defaults to False.
        simplify : bool, optional
            Whether to simplify the weights matrix to a single row if all rows 
            are identical. Defaults to False.
        normalize : bool, optional
            Whether to normalize the weights so that each row sums to 1. 
            Defaults to False.

        Returns
        -------
        FloatMatrix
            A 2d array of weights with shape (n_samples, n_pix).
        """
        flux = self._getFlux(
            bg_flux=bg_flux,
            ensure_not_negative=ensure_not_negative,
            simplify=False,
        ).copy()
        mask = self._getMask(
            centre=centre,
            from_model=from_model,
            simplify=False,
        )[:flux.shape[0],:]

        flux[~mask] = 0.0

        if weigh_by_bin_size:
            flux *= self.dx[None,:]
        if normalize:
            flux /= flux.sum(axis=1)[:,None]
        if simplify and _can_simplify_matrix(flux):
            return flux[:1]
        return flux

    @lru_cache
    def _getNWeights(
        self,
        *,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
        ensure_not_negative: bool = False,
    ) -> int:
        """
        Get the number of rows in the matrix returned by `getFluxWeights`
        for the given parameters if `simplify` were True.

        Parameters
        ----------
        centre : Literal["wave", "centroid", "median", "mode"], optional
            See `getMask` for details. Defaults to "wave".
        from_model : bool, optional
            See `getMask` for details. Defaults to True.
        bg_flux : BackgroundFlux | None, optional
            See `getFlux` for details. Defaults to None.
        ensure_not_negative : bool, optional
            See `getFlux` for details. Defaults to False.
        
        Returns
        -------
        int
            The number of rows in the matrix returned by `getFluxWeights`.
        """
        n_flux = self._getFlux(
            bg_flux=bg_flux,
            ensure_not_negative=ensure_not_negative,
            simplify=True,
        ).shape[0]
        n_mask = self._getMask(
            centre=centre,
            from_model=from_model,
            simplify=True,
        ).shape[0]
        return max(n_flux, n_mask)

    @lru_cache
    def _getMomentParams(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
    ) -> MomentParams:
        """
        Get a MomentParams object for the given parameters.

        Parameters
        ----------
        from_model : bool, optional
            Whether to use the model or data for measurements. Defaults to True.
        bg_flux : BackgroundFlux | None, optional
            The background fluxes to exclude. If None, defaults to excluding all
            background fluxes excelt line emission.
        centre : Literal["wave", "centroid", "median", "mode"], optional
            See `getMask` for details. Defaults to "wave".

        Returns
        -------
        MomentParams
            A MomentParams object containing the relevant parameters.

        Notes
        -----
        MomentParams is a frozen dataclass, so caching and reusing the output of
        this method is safe and efficient.
        """
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})
        return MomentParams(
            self,
            from_model=from_model,
            bg_flux=bg_flux,
            centre=centre,
        )

    ### Measures

    def getOut(
        self, 
        unit: UnitType | CompositeUnitType | None = None,
    ) -> MeasureResult:
        """Get a clean MeasureResult object."""
        return MeasureResult.create_empty(self.n_samples, unit=unit)

    ###

    @property
    def _wave_unit(self) -> CompositeUnitType:
        return self.info.units.wavelength_unit

    @property
    def _flux_unit(self) -> CompositeUnitType:
        return self.info.units.flux_unit

    @property
    def _int_flux_unit(self) -> CompositeUnitType:
        return self.info.units.strength_unit

    @property
    def _kms_unit(self) -> CompositeUnitType:
        return Unit("1.0 km/s")

    ### Public methods

    @wraps(_getMask)
    @pydantic.validate_call
    def getMask(
        self,
        *,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
        from_model: bool = True,
        simplify: bool = False,
    ) -> BoolMatrix:
        return self._getMask(
            centre=centre,
            from_model=from_model,
            simplify=simplify,
        )

    @wraps(_getFlux)
    @pydantic.validate_call
    def getFlux(
        self,
        *,
        bg_flux: BackgroundFlux | None = None,
        ensure_not_negative: bool = False,
        simplify: bool = False,
    ) -> FloatMatrix:
        return self._getFlux(
            bg_flux=bg_flux,
            ensure_not_negative=ensure_not_negative,
            simplify=simplify,
        )

    @wraps(_getFluxWeights)
    @pydantic.validate_call
    def getFluxWeights(
        self,
        *,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
        weigh_by_bin_size: bool = False,
        ensure_not_negative: bool = False,
        simplify: bool = False,
        normalize: bool = False,
    ) -> FloatMatrix:
        return self._getFluxWeights(
            centre=centre,
            from_model=from_model,
            bg_flux=bg_flux,
            weigh_by_bin_size=weigh_by_bin_size,
            ensure_not_negative=ensure_not_negative,
            simplify=simplify,
            normalize=normalize,
        )

    @wraps(_getNWeights)
    @pydantic.validate_call
    def getNWeights(
        self,
        *,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
        ensure_not_negative: bool = False,
    ) -> int:
        return self._getNWeights(
            centre=centre,
            from_model=from_model,
            bg_flux=bg_flux,
            ensure_not_negative=ensure_not_negative,
        )

    @wraps(_getMomentParams)
    @pydantic.validate_call
    def getMomentParams(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
        centre: Literal["wave", "centroid", "median", "mode"] = "wave",
    ) -> MomentParams:
        return self._getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre=centre,
        )
