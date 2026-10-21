from functools import wraps

import pydantic
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.decorators import lru_cache

from ..measure_result import FrozenMeasureResult
from .base_analyser import _BaseAnalyser


@pydantic.dataclasses.dataclass
class DistributionAnalyser(_BaseAnalyser):
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
    def _measureMAD(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the Median Absolute Deviation (MAD).
        """
        if bg_flux is None:
            return self._measureMAD(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        out = self.getOut(self._wave_unit)
        median = self._measureQuantile(
            0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        if from_model:
            from ...measures.exact import get_median_absolute_deviation

            n_em = 1 if self.v_profile_is_fixed else self.n_em

            mask = self._getMask(
                centre="median",
                from_model=True,
                simplify=False,
            )[:n_em]

            args_iter = zip(
                median.val[:n_em],
                (sample.em.cdf for sample in self.error_result),
                (self.x[m] for m in mask),
            )
            out.update_with_single_values(
                list(self.starmap(get_median_absolute_deviation, args_iter)),
                index=slice(None, n_em),
            )
        else:
            from ...measures.approx import get_median_absolute_deviation

            n = self._getNWeights(
                centre="median",
                from_model=True,
                bg_flux=bg_flux,
            )
            flux = self._getFlux(
                bg_flux=bg_flux,
            )[:n]
            mask = self._getMask(
                centre="median",
                from_model=True,
            )[:n]

            cens = median.val[:n]
            xs = (self.x[m] for m in mask)
            ys = (y[m] for y, m in zip(flux, mask))
            dxs = (self.dx[m] for m in mask)
            args_iter = zip(cens, xs, ys, dxs)

            out.update_with_single_values(
                list(self.starmap(get_median_absolute_deviation, args_iter)),
                index=slice(None, n),
            )

        return out.freeze()

    def _measureMADVelocity(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the velocity Median Absolute Deviation (MAD) w.r.t. the 
        median.
        """
        if bg_flux is None:
            return self._measureMADVelocity(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        median = self._measureQuantile(
            0.5,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        mad = self._measureMAD(
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (mad / median).to_c()

    @lru_cache
    def _measureAAD(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the Average Absolute Deviation (AAD).
        """
        if bg_flux is None:
            return self._measureAAD(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        out = self.getOut(self._wave_unit)
        centroid = self._measureFirstMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )
        if from_model:
            from ...measures.exact import get_average_absolute_deviation

            n_em = 1 if self.v_profile_is_fixed else self.n_em

            args_iter = zip(
                centroid.val[:n_em],
                (sample.em.dist for sample in self.error_result),
                (sample.em.weight for sample in self.error_result),
            )
            out.update_with_single_values(
                list(self.starmap(get_average_absolute_deviation, args_iter)),
                index=slice(None, n_em),
            )
        else:
            from ...measures.approx import get_average_absolute_deviation

            n = self._getNWeights(
                centre="centroid",
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
                centre="centroid",
                from_model=True,
                simplify=False,
            )[:n]

            cens = centroid.val[:n]
            xs = (self.x[m] for m in mask)
            ys = (y[m] for y, m in zip(flux, mask))
            dxs = (self.dx[m] for m in mask)
            args_iter = zip(cens, xs, ys, dxs)

            out.update_with_single_values(
                list(self.starmap(get_average_absolute_deviation, args_iter)),
                index=slice(None, n),
            )

        return out.freeze()

    def _measureAADVelocity(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the velocity Average Absolute Deviation (AAD) w.r.t. the 
        centroid.
        """
        if bg_flux is None:
            return self._measureAADVelocity(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        centroid = self._measureFirstMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )
        aad = self._measureAAD(
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (aad / centroid).to_c()

    ### Public methods

    @wraps(_measureMAD)
    @pydantic.validate_call
    def measureMAD(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureMAD(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureMADVelocity)
    @pydantic.validate_call
    def measureMADVelocity(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureMADVelocity(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureAAD)
    @pydantic.validate_call
    def measureAAD(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureAAD(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureAADVelocity)
    @pydantic.validate_call
    def measureAADVelocity(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureAADVelocity(
            from_model=from_model,
            bg_flux=bg_flux,
        )
