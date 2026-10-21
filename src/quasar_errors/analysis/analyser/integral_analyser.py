from functools import wraps
from itertools import repeat
from typing import Literal

import numpy as np
import pydantic
from quasar_typing.bounds import CoordBounds
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.decorators import lru_cache

from ..measure_result import FrozenMeasureResult
from .base_analyser import _BaseAnalyser


@pydantic.dataclasses.dataclass
class IntegralAnalyser(_BaseAnalyser):
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
    def _measureIntegratedFlux(
        self,
        x_bounds: CoordBounds,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Measure the integrated flux over a given range.
        """
        if bg_flux is None:
            return self._measureIntegratedFlux(
                x_bounds,
                from_model=from_model,
                bg_flux=BackgroundFlux(set()),
            )

        out = self.getOut(self._int_flux_unit)
        mask = (x_bounds[0] <= self.x) & (self.x <= x_bounds[1])
        if not mask.any():
            return out

        dxs = self.dx[mask][None,:]
        if from_model:
            ys = 0
            n = 0
            if "pl" not in bg_flux and self.n_pl > 0:
                n = max(n, self.n_pl)
                ys += self.y_pl[:,mask]
            if "fe" not in bg_flux and self.n_fe > 0:
                n = max(n, self.n_fe)
                ys += self.y_fe[:,mask]
            if "ba" not in bg_flux and self.n_ba > 0:
                n = max(n, self.n_ba)
                ys += self.y_ba[:,mask]
            if "hg" not in bg_flux and self.n_hg > 0:
                n = max(n, self.n_hg)
                ys += self.y_hg[:,mask]
            if "em" not in bg_flux and self.n_em > 0:
                n = max(n, self.n_em)
                ys += self.y_em[:,mask]

            out.update_with_single_values(
                (ys * dxs).sum(axis=1),
                index=slice(None, n),
            )
        else:
            from ..spectrum_inner import total

            ys = self._getFlux(
                bg_flux=bg_flux,
                ensure_not_negative=False,
                simplify=True,
            )[:,mask]
            dys = self.dy[mask][None,:]

            out.update_with_single_values(
                total(ys, dxs, dys),
                index=slice(None, ys.shape[0]),
            )

        return out.freeze()

    @lru_cache
    def _measureAverageFluxDensity(
        self,
        x_bounds: CoordBounds,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Measure the average flux density over a given range.
        """
        if bg_flux is None:
            return self._measureAverageFluxDensity(
                x_bounds,
                from_model=from_model,
                bg_flux=BackgroundFlux(set()),
            )
        
        mask = (x_bounds[0] <= self.x) & (self.x <= x_bounds[1])
        n = int(mask.sum())
        if n == 0:
            return self.getOut(self._flux_unit).freeze()

        delta_x: float = self.dx[mask].sum()
        integral = self._measureIntegratedFlux(
            x_bounds=x_bounds,
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return integral / (delta_x * self._wave_unit)

    @lru_cache
    def _measureComponentFlux(
        self,
        x: float,
        *,
        component: Literal["pl", "fe", "ba", "hg", "em"],
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        out = self.getOut(self._flux_unit)
        from_model |= (x < self.x[0]) | (self.x[-1] < x)

        if from_model:
            n = getattr(self, f"n_{component}")
            if n == 0:
                return out.freeze()
            
            out.update_with_single_values(
                [
                    getattr(s, component) 
                    for _, s in zip(range(n), self.error_result)
                ],
                index=slice(None, n),
            )
        else:
            from ...measures.approx import u_yinterp
            ys = self._getFlux(
                bg_flux=BackgroundFlux({"all", component}),
                ensure_not_negative=False,
                simplify=True,
            )
            out.update_with_single_values(
                [u_yinterp(x, self.x, y, self.dy) for y in ys],
                index=slice(None, ys.shape[0]),
            )

        return out.freeze()

    def _measurePLFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureComponentFlux(
            x,
            component="pl",
            from_model=from_model,
        )

    def _measureFEFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureComponentFlux(
            x,
            component="fe",
            from_model=from_model,
        )

    def _measureBAFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureComponentFlux(
            x,
            component="ba",
            from_model=from_model,
        )

    def _measureHGFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureComponentFlux(
            x,
            component="hg",
            from_model=from_model,
        )

    def _measureEMFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureComponentFlux(
            x,
            component="em",
            from_model=from_model,
        )

    @lru_cache
    def _measureEW(
        self,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        """
        Measure the equivalent width (EW).
        """
        out = self.getOut(self._wave_unit)
        if from_model:
            n = max(self.n_em, self.n_pl)

            mask = self._getMask(
                centre="wave",
                from_model=True,
                simplify=False,
            )[:n]

            xs = (self.x[m] for m in mask)
            dxs = (self.dx[m] for m in mask)

            if self.n_pl < n:
                # Assume: constant power law continuum
                ems = (sample.em for sample in self.error_result)
                pls = repeat(self.error_result[0].pl, n)
            elif self.n_em < n:
                # Assume: constant emission line models
                pls = (sample.pl for sample in self.error_result)
                ems = repeat(self.error_result[0].em, n)
            else:
                ems = (sample.em for sample in self.error_result)
                pls = (sample.pl for sample in self.error_result)

            weights = (em(x) * dx for em, x, dx in zip(ems, xs, dxs))
            inv_continua = (1 / pl(x) for pl, x in zip(pls, xs))
        else:
            bg_flux = BackgroundFlux({"all", "em"})
            n_em = self._getNWeights(
                centre="wave",
                from_model=True,
                bg_flux=bg_flux,
                ensure_not_negative=False,
            )
            flux = self._getFlux(
                bg_flux=bg_flux,
                ensure_not_negative=False,
                simplify=False,
            )[:n_em]
            mask = self._getMask(
                centre="wave",
                from_model=True,
                simplify=False,
            )[:n_em]

            n = max(n_em, self.n_pl)
            weights = np.empty((n, self.n_pix), dtype=np.float64)
            inv_continua = np.empty((n, self.n_pix), dtype=np.float64)
            if self.n_pl < n:
                # Assume: constant power law continuum
                weights[:] = np.where(mask, flux, 0.0) * self.dx[None,:]
                inv_continua[:] = 1 / self.y_pl[:1,:]
            elif self.n_em < n:
                # Assume: constant emission line models
                weights[:] = np.where(mask[:1,:], flux[:1,:], 0.0) * self.dx[None,:]
                inv_continua[:] = 1 / self.y_pl
            else:
                weights[:] = np.where(mask, flux, 0.0) * self.dx[None,:]
                inv_continua[:] = 1 / self.y_pl

        out.update_with_single_values(
            [
                np.dot(weights_i, inv_continua_i) 
                for weights_i, inv_continua_i in zip(weights, inv_continua)
            ],
            index=slice(None, n),
        )
        return out.freeze()

    ### Public methods

    @wraps(_measureIntegratedFlux)
    @pydantic.validate_call
    def measureIntegratedFlux(
        self,
        x_bounds: CoordBounds,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureIntegratedFlux(
            x_bounds,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureAverageFluxDensity)
    @pydantic.validate_call
    def measureAverageFlux(
        self,
        x_bounds: CoordBounds,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureAverageFluxDensity(
            x_bounds,
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureComponentFlux)
    @pydantic.validate_call
    def measureComponentFlux(
        self,
        x: float,
        *,
        component: Literal["pl", "fe", "ba", "hg", "em"],
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureComponentFlux(
            x,
            component=component,
            from_model=from_model,
        )

    @wraps(_measurePLFlux)
    @pydantic.validate_call
    def measurePLFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measurePLFlux(
            x,
            from_model=from_model,
        )

    @wraps(_measureFEFlux)
    @pydantic.validate_call
    def measureFEFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureFEFlux(
            x,
            from_model=from_model,
        )

    @wraps(_measureBAFlux)
    @pydantic.validate_call
    def measureBAFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureBAFlux(
            x,
            from_model=from_model,
        )

    @wraps(_measureHGFlux)
    @pydantic.validate_call
    def measureHGFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureHGFlux(
            x,
            from_model=from_model,
        )

    @wraps(_measureEMFlux)
    @pydantic.validate_call
    def measureEMFlux(
        self,
        x: float,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureEMFlux(
            x,
            from_model=from_model,
        )

    @wraps(_measureEW)
    @pydantic.validate_call
    def measureEW(
        self,
        *,
        from_model: bool = True,
    ) -> FrozenMeasureResult:
        return self._measureEW(
            from_model=from_model,
        )
