from functools import wraps
from itertools import repeat

import pydantic
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.decorators import lru_cache

from ..measure_result import FrozenMeasureResult
from .base_analyser import _BaseAnalyser


@pydantic.dataclasses.dataclass
class DiscreteAnalyser(_BaseAnalyser):
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
    @lru_cache
    def _measurePeak(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> tuple[FrozenMeasureResult, FrozenMeasureResult]:
        """
        Calculate the peak flux density value and its location.
        """
        if bg_flux is None:
            return self._measurePeak(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        modes = self.getOut(self._wave_unit)
        maxs = self.getOut(self._flux_unit)
        if from_model:
            from ...measures.exact import get_peak
            
            if not self.n_em:
                return (modes, maxs)

            mask = self._getMask(
                centre="wave",
                from_model=True,
                simplify=False,
            )[:self.n_em]

            args_iter = zip(
                (sample.em for sample in self.error_result),
                (self.x[m] for m in mask),
            )
            results = list(self.starmap(get_peak, args_iter))
        else:
            from ...measures.approx import get_peak

            n = self._getNWeights(
                centre="wave",
                from_model=True,
                bg_flux=bg_flux,
                ensure_not_negative=False,
            )
            ys = self._getFlux(
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
                (self.x[m] for m in mask),
                (y[m] for y, m in zip(ys, mask)),
                (self.dx[m] for m in mask),
            )
            results = list(self.starmap(get_peak, args_iter))

        modes.update_with_single_values(
            [r[0] for r in results],
            index=slice(None, len(results)),
        )
        maxs.update_with_single_values(
            [r[1] for r in results],
            index=slice(None, len(results)),
        )
        return (modes.freeze(), maxs.freeze())

    def _measureMode(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the mode of the flux density distribution.
        """
        if bg_flux is None:
            return self._measureMode(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        return self._measurePeak(
            from_model=from_model,
            bg_flux=bg_flux,
        )[0]

    def _measureModeOffset(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the velocity offset (kms) of the mode w.r.t. the central 
        wavelength.
        """
        if bg_flux is None:
            return self._measureModeOffset(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        mode = self._measurePeak(
            from_model=from_model,
            bg_flux=bg_flux,
        )[0]
        return (mode / (self.wave * self._wave_unit) - 1).to_c()
    
    def _measureMax(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the maximum flux density value.
        """
        if bg_flux is None:
            return self._measureMax(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        return self._measurePeak(
            from_model=from_model,
            bg_flux=bg_flux,
        )[1]

    @lru_cache
    def _measureLeftEdge(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the left edge of the FWHM at the given height.
        """
        if bg_flux is None:
            return self._measureLeftEdge(
                h=h,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        out = self.getOut(self._wave_unit)
        mode, peak = self._measurePeak(
            from_model=from_model,
            bg_flux=bg_flux,
        )
        if from_model:
            from ...measures.exact import get_left_edge_at_height

            if self.v_profile_is_fixed:
                n_em = 1
            elif self.stacked_em_params or self.em_params:
                n_em = self.n_em
            else:
                return out

            mask = self._getMask(
                centre="wave",
                from_model=True,
                simplify=False,
            )[:n_em]

            args_iter = zip(
                (sample.em for sample in self.error_result),
                (self.x[m] for m in mask),
                mode.val[:n_em],
                h * peak.val[:n_em],
                repeat(self.fwhm_strategy),
            )
            out.update_with_single_values(
                list(self.starmap(get_left_edge_at_height, args_iter)),
                index=slice(None, n_em),
            )
        else:
            from ...measures.approx import get_left_edge_at_height

            moment_params = self._getMomentParams(
                from_model=False,
                bg_flux=bg_flux,
                centre="mode",
            )
            n = self._getNWeights(
                centre="mode",
                from_model=False,
                bg_flux=bg_flux,
                ensure_not_negative=False,
            )
            mask = moment_params.mask[:n]

            args_iter = zip(
                (self.x[m] for m in mask), 
                (y[m] for y, m in zip(moment_params.ys, mask)), 
                (self.dy[m] for m in mask), 
                mode.val[:n], 
                h * peak.val[:n],
                repeat(self.fwhm_strategy),
            )
            out.update_with_list_of_tuples(
                list(self.starmap(get_left_edge_at_height, args_iter)),
                index=slice(None, n),
            )

        return out.freeze()

    @lru_cache
    def _measureRightEdge(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the right edge of the FWHM at the given height.
        """
        if bg_flux is None:
            return self._measureRightEdge(
                h=h,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        out = self.getOut(self._wave_unit)
        mode, peak = self._measurePeak(
            from_model=from_model,
            bg_flux=bg_flux,
        )
        if from_model:
            from ...measures.exact import get_right_edge_at_height
            
            if self.v_profile_is_fixed:
                n_em = 1
            elif self.stacked_em_params or self.em_params:
                n_em = self.n_em
            else:
                return out

            mask = self._getMask(
                centre="wave",
                from_model=from_model,
                simplify=False,
            )[:n_em]

            args_iter = zip(
                (sample.em for sample in self.error_result),
                (self.x[m] for m in mask),
                mode.val[:n_em],
                h * peak.val[:n_em],
                repeat(self.fwhm_strategy),
            )
            out.update_with_single_values(
                list(self.starmap(get_right_edge_at_height, args_iter)),
                index=slice(None, n_em),
            )
        else:
            from ...measures.approx import get_right_edge_at_height

            moment_params = self._getMomentParams(
                from_model=False,
                bg_flux=bg_flux,
                centre="mode",
            )
            n = self._getNWeights(
                centre="mode",
                from_model=False,
                bg_flux=bg_flux,
                ensure_not_negative=False,
            )
            mask = moment_params.mask[:n]

            args_iter = zip(
                (self.x[m] for m in mask), 
                (y[m] for y, m in zip(moment_params.ys, mask)), 
                (self.dy[m] for m in mask), 
                mode.val[:n], 
                h * peak.val[:n],
                repeat(self.fwhm_strategy),
            )
            out.update_with_list_of_tuples(
                list(self.starmap(get_right_edge_at_height, args_iter)),
                index=slice(None, n),
            )

        return out.freeze()

    def _measureFWHM(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the full width at the given height.
        """
        if bg_flux is None:
            return self._measureFWHM(
                h=h,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        left_edge = self._measureLeftEdge(
            h=h, from_model=from_model, bg_flux=bg_flux,
        )
        right_edge = self._measureRightEdge(
            h=h, from_model=from_model, bg_flux=bg_flux,
        )
        fwhm = right_edge - left_edge

        return fwhm

    def _measureLeftEdgeVelocity(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the full velocity width (km/s) at the given height.
        """
        if bg_flux is None:
            return self._measureLeftEdgeVelocity(
                h=h,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        mode = self._measurePeak(
            from_model=from_model,
            bg_flux=bg_flux,
        )[0]
        left_edge = self._measureLeftEdge(
            h=h,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (left_edge / mode).to_c()

    def _measureRightEdgeVelocity(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the full velocity width (km/s) at the given height.
        """
        if bg_flux is None:
            return self._measureRightEdgeVelocity(
                h=h,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        mode = self._measurePeak(
            from_model=from_model,
            bg_flux=bg_flux,
        )[0]
        right_edge = self._measureRightEdge(
            h=h,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (right_edge / mode).to_c()

    
    def _measureFWHMVelocity(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the full velocity width (km/s) at the given height.
        """
        if bg_flux is None:
            return self._measureFWHMVelocity(
                h=h,
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        mode = self._measurePeak(
            from_model=from_model,
            bg_flux=bg_flux,
        )[0]
        fwhm = self._measureFWHM(
            h=h,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (fwhm / mode).to_c()

    @lru_cache
    def _measureWhittleK1(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the Whittle K1 parameter: 
            K1 = 1.524 * (FWHM / FW20)
        """
        if bg_flux is None:
            return self._measureWhittleK1(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        fwhm = self._measureFWHM(
            h=0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        fw20 = self._measureFWHM(
            h=0.2,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return 1.524 * fwhm / fw20

    def _measureWhittleK2(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the Whittle K2 parameter:
            K2 = 1.397 * (FWHM / IPW(10%))
        """
        if bg_flux is None:
            return self._measureWhittleK2(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        ipw10 = self._measureIPWidth(
            10,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        fwhm = self._measureFWHM(
            h=0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return 1.397 * fwhm / ipw10

    ### Public API

    @wraps(_measurePeak)
    @pydantic.validate_call
    def measurePeak(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> tuple[FrozenMeasureResult, FrozenMeasureResult]:
        return self._measurePeak(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureMode)
    @pydantic.validate_call
    def measureMode(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureMode(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureModeOffset)
    @pydantic.validate_call
    def measureModeOffset(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureModeOffset(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureMax)
    @pydantic.validate_call
    def measureMax(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureMax(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureLeftEdge)
    @pydantic.validate_call
    def measureLeftEdge(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureLeftEdge(
            h=h,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureRightEdge)
    @pydantic.validate_call
    def measureRightEdge(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureRightEdge(
            h=h,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureFWHM)
    @pydantic.validate_call
    def measureFWHM(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureFWHM(
            h=h,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureLeftEdgeVelocity)
    @pydantic.validate_call
    def measureLeftEdgeVelocity(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureLeftEdgeVelocity(
            h=h,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureRightEdgeVelocity)
    @pydantic.validate_call
    def measureRightEdgeVelocity(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureRightEdgeVelocity(
            h=h,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureFWHMVelocity)
    @pydantic.validate_call
    def measureFWHMVelocity(
        self,
        *,
        h: float = 0.5,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureFWHMVelocity(
            h=h,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureWhittleK1)
    @pydantic.validate_call
    def measureWhittleK1(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureWhittleK1(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureWhittleK2)
    @pydantic.validate_call
    def measureWhittleK2(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureWhittleK2(
            from_model=from_model,
            bg_flux=bg_flux,
        )
