from functools import wraps
from itertools import repeat

from pydantic import validate_call
from pydantic.dataclasses import dataclass
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.decorators import lru_cache

from ..measure_result import FrozenMeasureResult
from .base_analyser import _BaseAnalyser


@dataclass
class QuantilesAnalyser(_BaseAnalyser):
    """
    Analyser for calculating quantiles of emission line profiles, as well as 
    measures derived from them.

    The base method is:

    - `measureQuantile`: calculates the wavelength value corresponding to a given quantile.

    The derived measures are:

    - `measureMedian`
    - `measureMedianOffset`
    - `measureIPBounds`
    - `measureIPWidth`
    - `measureIPVelocity`
    - `measureWhittleBounds`
    - `measureWhittleA`
    - `measureWhittleB`
    - `measureWhittleShift`
    - `measureWhittleAsymmetry`

    Notes
    -----
    Private methods are prefixed with an underscore (_) and are intended for 
    internal use within the class, or use by an experienced developer. These 
    methods are sometimes cached (LRU). 

    External methods have no underscore prefix and are intended for the general 
    user. They are simple wrappers of their corresponding private methods using 
    `pydantic.validate_call`.
    """

    ### Public API

    @validate_call
    def measureQuantile(
        self,
        q: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the wavelength value at the corresponding quantile, `q`.
        """
        return self._measureQuantile(
            q,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureMedian(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the wavelength value at the median (0.5 quantile).
        """
        return self._measureMedian(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureMedianOffset(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the velocity offset (km/s) of the median w.r.t. the centre.
        """
        return self._measureMedianOffset(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureIPBounds(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> tuple[FrozenMeasureResult, FrozenMeasureResult]:
        """
        Calculate the inter-percentile wavelength bounds.
        """
        return self._measureIPBounds(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureIPWidth(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the inter-percentile wavelength width.
        """
        return self._measureIPWidth(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureIPVelocity(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the inter-percentile velocity width (km/s).
        """
        return self._measureIPVelocity(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureWhittleBounds(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> tuple[FrozenMeasureResult, FrozenMeasureResult]:
        """
        Calculate the Whittle velocity bounds (km/s).
        """
        return self._measureWhittleBounds(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureWhittleA(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the Whittle 'a' bound (km/s).
        """
        return self._measureWhittleA(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureWhittleB(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the Whittle 'b' bound (km/s).
        """
        return self._measureWhittleB(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureWhittleShift(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the Whittle shift parameter (km/s): S = a - b.
        """
        return self._measureWhittleShift(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @validate_call
    def measureWhittleAsymmetry(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the Whittle asymmetry parameter: A = (a - b) / (a + b).
        """
        return self._measureWhittleAsymmetry(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    ### Private API

    @wraps(measureQuantile)
    @lru_cache
    def _measureQuantile(
        self,
        q: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        if bg_flux is None:
            bg_flux = BackgroundFlux({"all", "em"})

        out = self.getOut(self._wave_unit)
        if from_model:
            from quasar_errors.measures.exact import get_quantile

            if self.v_profile_is_fixed:
                n_em = 1
            elif self.stacked_em_params or self.em_params:
                n_em = self.n_em
            else:
                return out.freeze()

            args_iter = zip(
                repeat(q, n_em),
                (sample.em.cdf for sample in self.error_result),
                repeat(self.x, n_em),
            )
            index = slice(None, n_em)
        else:
            from quasar_errors.measures.approx import get_quantile

            # Check if simplification is possible
            n = self._getNWeights(
                centre="wave",
                from_model=True,
                bg_flux=bg_flux,
                ensure_not_negative=False,
            )
            flux = self._getFlux(
                bg_flux=bg_flux,
                ensure_not_negative=False,
                simplify=False,
            )[:n]
            mask = self._getMask(
                centre="wave",
                from_model=True,
                simplify=False,
            )[:n]
            args_iter = zip(
                repeat(q, n), 
                (self.x[m] for m in mask), 
                (y[m] for y, m in zip(flux, mask)), 
                (self.dx[m] for m in mask),
            )
            index = slice(None, n)

        out.update_with_single_values(
            list(self.starmap(get_quantile, args_iter)),
            index=index,
        )
        return out.freeze()


    @wraps(measureMedian)
    def _measureMedian(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        if bg_flux is None:
            return self._measureQuantile(
                0.5,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        return self._measureQuantile(
            0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )


    @wraps(measureMedianOffset)
    def _measureMedianOffset(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        if bg_flux is None:
            return self._measureMedianOffset(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        median = self._measureQuantile(
            0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (median / (self.wave * self._wave_unit) - 1).to_c()


    @wraps(measureIPBounds)
    def _measureIPBounds(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> tuple[FrozenMeasureResult, FrozenMeasureResult]:
        if bg_flux is None:
            return self._measureIPBounds(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        func = self._measureQuantile
        return (
            func(p / 200, from_model=from_model, bg_flux=bg_flux),
            func(1 - p / 200, from_model=from_model, bg_flux=bg_flux),
        )


    @wraps(measureIPWidth)
    def _measureIPWidth(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        if bg_flux is None:
            return self._measureIPWidth(
                p,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        lower, upper = self._measureIPBounds(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return upper - lower


    @wraps(measureIPVelocity)
    def _measureIPVelocity(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the inter-percentile velocity width (km/s).
        """
        if bg_flux is None:
            return self._measureIPVelocity(
                p,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        ipw = self._measureIPWidth(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        median = self._measureQuantile(
            0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (ipw / median).to_c()
    
    
    @wraps(measureWhittleBounds)
    def _measureWhittleBounds(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> tuple[FrozenMeasureResult, FrozenMeasureResult]:
        if bg_flux is None:
            return self._measureWhittleBounds(
                p,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        lower = self._measureQuantile(
            p / 200, from_model=from_model, bg_flux=bg_flux,
        )
        upper = self._measureQuantile(
            1 - p / 200, from_model=from_model, bg_flux=bg_flux,
        )
        m = self._measureQuantile(
            0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (
            abs(lower / m - 1).to_c(), 
            abs(upper / m - 1).to_c(),
        )
    
    
    @wraps(measureWhittleA)
    def _measureWhittleA(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        if bg_flux is None:
            return self._measureWhittleA(
                p,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        lower = self._measureQuantile(
            p / 200, from_model=from_model, bg_flux=bg_flux,
        )
        m = self._measureQuantile(
            0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return abs(lower / m - 1).to_c()


    @wraps(measureWhittleB)
    def _measureWhittleB(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        if bg_flux is None:
            return self._measureWhittleB(
                p,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        upper = self._measureQuantile(
            1 - p / 200, from_model=from_model, bg_flux=bg_flux,
        )
        m = self._measureQuantile(
            0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return abs(upper / m - 1).to_c()


    @wraps(measureWhittleShift)
    def _measureWhittleShift(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        if bg_flux is None:
            return self._measureWhittleShift(
                p,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        a, b = self._measureWhittleBounds(
            p,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return a - b


    @wraps(measureWhittleAsymmetry)
    @lru_cache
    def _measureWhittleAsymmetry(
        self,
        p: float,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        from numpy import hypot

        if bg_flux is None:
            return self._measureWhittleAsymmetry(
                p,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        a, b = self._measureWhittleBounds(
            p,
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
        return out.freeze()