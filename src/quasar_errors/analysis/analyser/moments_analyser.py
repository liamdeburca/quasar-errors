from functools import wraps

import numpy as np
import pydantic
from quasar_typing.string_selections import BackgroundFlux
from quasar_utils.decorators import lru_cache

from ..measure_result import FrozenMeasureResult
from .base_analyser import _BaseAnalyser


@pydantic.dataclasses.dataclass
class MomentsAnalyser(_BaseAnalyser):
    """
    Analyser for calculating statistical moment of emission line profiles, as 
    well as measures derived from them.

    The moments are:

    - 0th moment: `measureZerothMoment`
    - 1st moment: `measureFirstMoment`
    - 2nd central moment: `measureSecondCentralMoment`
    - 3rd central moment: `measureThirdCentralMoment`
    - 4th central moment: `measureFourthCentralMoment`

    The derived measures are:

    - Centroid: `measureCentroid`, proxy for `measureFirstMoment`
    - Mean offset: `measureMeanOffset`, derived from the 1st moment
    - Dispersion: `measureDispersion`, derived from the 2nd central moment
    - Velocity dispersion: `measureVelocityDispersion`, derived from the 1st and 
    2nd central moments
    - Skewness: `measureSkewness`, derived from the 2nd and 3rd central moments
    - Kurtosis: `measureKurtosis`, derived from the 2nd and 4th central moments
    - Strength: `measureStrength`, proxy for `measureZerothMoment`

    Notes
    -----
    Private methods are prefixed with an underscore (_) and are intended for 
    internal use within the class, or use by an experienced developer. These 
    methods are sometimes cached (LRU). 

    External methods have no underscore prefix and are intended for the general 
    user. They are simple wrappers of their corresponding private methods using 
    `pydantic.validate_call`.
    """

    ### Moments

    @lru_cache
    def _measureZerothMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the zeroth moment.
        """
        if bg_flux is None:
            return self._measureZerothMoment(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        out = self.getOut(self._int_flux_unit)
        if from_model:
            if self.em_params is None:
                return out.freeze()

            n_em = 1 if self.v_profile_is_fixed else self.n_em
            d = self.stacked_em_params or self.em_params

            out.update_with_single_values(
                [s.sum() for s in d["strengths"][:n_em]],
                index=slice(None, n_em),
            )
        else:
            from ..spectrum_inner import total

            moment_params = self._getMomentParams(
                from_model=False,
                bg_flux=bg_flux,
                centre="wave",
            )
            mask = moment_params.mask
            ys = np.where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and np.allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = np.where(mask, moment_params.dys, 0)

            out.update(
                total(ys, moment_params.dxs, dys),
                index=slice(None, ys.shape[0]),
            )
            # Correct if IPV bounds are used
            if self.mask_correction != 1.0:
                out *= self.mask_correction

        return out.freeze()

    @lru_cache
    def _measureFirstMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the first moment.
        """
        if bg_flux is None:
            return self._measureFirstMoment(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        out = self.getOut(self._wave_unit)
        if from_model:
            if self.em_params is None:
                return out.freeze()

            n_em = 1 if self.v_profile_is_fixed else self.n_em
            d = self.stacked_em_params or self.em_params

            mus = d["mus"][:n_em]
            norm_strengths = d["norm_strengths"][:n_em]
            args_iter = zip(norm_strengths, mus)

            out.update_with_single_values(
                list(self.starmap(np.dot, args_iter)),
                index=slice(None, n_em),
            )
        else:
            from ..spectrum_inner import nth_moment

            moment_params = self._getMomentParams(
                from_model=from_model,
                bg_flux=bg_flux,
                centre="wave",
            )
            mask = moment_params.mask
            ys = np.where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and np.allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = np.where(mask, moment_params.dys, 0)

            _out = nth_moment(
                1,
                moment_params.xs, ys,
                moment_params.dxs, dys,
                tot=moment_params.tot,
            )

            out.update(_out, index=slice(None, ys.shape[0]))
            out.remove_negatives()

        return out.freeze()

    @lru_cache
    def _measureSecondCentralMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the second central moment.
        """
        if bg_flux is None:
            return self._measureSecondCentralMoment(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        out = self.getOut(self._wave_unit ** 2)
        moment_params = self._getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )
        m1 = moment_params.m1

        if from_model:
            from ..inner import weighted_second_moment

            if self.em_params is None:
                return out.freeze()

            n_em = 1 if self.v_profile_is_fixed else self.n_em
            d = self.stacked_em_params or self.em_params

            weights = d["norm_strengths"][:n_em]
            first_moments = m1.val[:n_em]
            mus = d["mus"][:n_em]
            sigmas = d["sigmas"][:n_em]
            args_iter = zip(weights, first_moments, mus, sigmas)

            out.update_with_single_values(
                list(self.starmap(weighted_second_moment, args_iter)),
                index=slice(None, n_em),
            )
        else:
            from ..spectrum_inner import nth_central_moment

            mask = moment_params.mask
            ys = np.where(mask, moment_params.ys, 0)
            if (ys.shape[0] > 1) and np.allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = np.where(mask, moment_params.dys, 0)
            _out = nth_central_moment(
                2,
                moment_params.xs, ys,
                moment_params.dxs, dys,
                m1=m1,
                tot=moment_params.tot,
            )

            out.update(_out, index=slice(None, ys.shape[0]))
            out.remove_negatives()

        return out.freeze()

    @lru_cache
    def _measureThirdCentralMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the third central moment.
        """
        if bg_flux is None:
            return self._measureThirdCentralMoment(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        out = self.getOut(self._wave_unit ** 3)
        moment_params = self._getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )

        if from_model:
            from ..inner import weighted_third_moment

            if not self.em_params:
                return out.freeze()

            n_em = 1 if self.v_profile_is_fixed else self.n_em
            d = self.stacked_em_params or self.em_params

            args_iter = zip(
                d["norm_strengths"][:n_em],
                moment_params.m1.val[:n_em],
                moment_params.cm2.val[:n_em],
                d["mus"][:n_em],
                d["sigmas"][:n_em],
            )
            out.update_with_single_values(
                list(self.starmap(weighted_third_moment, args_iter)),
                index=slice(None, n_em),
            )
        else:
            from ..spectrum_inner import nth_central_moment

            mask = moment_params.mask
            ys = np.where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and np.allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = np.where(mask, moment_params.dys, 0)
            _out = nth_central_moment(
                3,
                moment_params.xs, ys,
                moment_params.dxs, dys,
                m1=moment_params.m1,
                tot=moment_params.tot,
            )

            out.update(_out, index=slice(None, ys.shape[0]))

        return out.freeze()

    @lru_cache
    def _measureFourthCentralMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the fourth central moment.
        """
        if bg_flux is None:
            return self._measureFourthCentralMoment(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        out = self.getOut(self._wave_unit ** 4)
        moment_params = self._getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )

        if from_model:
            from ..inner import weighted_fourth_moment

            if not self.em_params:
                return out.freeze()

            n_em = 1 if self.v_profile_is_fixed else self.n_em
            d = self.stacked_em_params or self.em_params

            args_iter = zip(
                d["norm_strengths"][:n_em],
                moment_params.m1.val[:n_em],
                moment_params.cm3.val[:n_em],
                d["mus"][:n_em],
                d["sigmas"][:n_em],
            )
            out.update_with_single_values(
                list(self.starmap(weighted_fourth_moment, args_iter)),
                index=slice(None, n_em),
            )
        else:
            from ..spectrum_inner import nth_central_moment

            mask = moment_params.mask
            ys = np.where(mask, moment_params.ys, 0)

            if (ys.shape[0] > 1) and np.allclose(ys[1:] - ys[:1], 0):
                mask = mask[:1]
                ys = ys[:1]

            dys = np.where(mask, moment_params.dys, 0)
            _out = nth_central_moment(
                4,
                moment_params.xs, ys,
                moment_params.dxs, dys,
                m1=moment_params.m1,
                tot=moment_params.tot,
            )

            out.update(_out, index=slice(None, ys.shape[0]))
            out.remove_negatives()

        return out.freeze()

    ### Derived Measures

    def _measureCentroid(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the mean / centroid / first moment.
        """
        if bg_flux is None:
            return self._measureCentroid(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        return self._measureFirstMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    def _measureCentroidOffset(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the offset of the mean / centroid / first moment.
        """
        if bg_flux is None:
            return self._measureCentroidOffset(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        m1 = self._measureFirstMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )
        return (m1 / (self.wave * self._wave_unit) - 1).to_c()

    def _measureDispersion(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the wavelength dispersion.
        """
        if bg_flux is None:
            return self._measureDispersion(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        return self._measureSecondCentralMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )**0.5

    @lru_cache
    def _measureVelocityDispersion(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the velocity dispersion w.r.t. the centroid.
        """
        if bg_flux is None:
            return self._measureVelocityDispersion(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        moment_params = self._getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )
        if from_model:
            return (moment_params.cm2**0.5 / moment_params.m1).to_c()

        from ..spectrum_inner import dispersion

        out = self.getOut()

        mask = moment_params.mask
        ys = np.where(mask, moment_params.ys, 0)

        if (ys.shape[0] > 1) and np.allclose(ys[1:] - ys[:1], 0):
            mask = mask[:1]
            ys = ys[:1]

        dys = np.where(mask, moment_params.dys, 0)
        _out = dispersion(
            moment_params.xs, ys,
            moment_params.dxs, dys,
            tot=moment_params.tot,
            m1=moment_params.m1,
            cm2=moment_params.cm2,
        )
        out.update(_out, index=slice(None, ys.shape[0]))

        return out.remove_negatives().freeze().to_c()

    @lru_cache
    def _measureSkewness(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculate the skewness.
        """
        if bg_flux is None:
            return self._measureSkewness(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        moment_params = self._getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )
        if from_model:
            return moment_params.cm3 / moment_params.cm2 ** (3 / 2)

        from ..spectrum_inner import skewness

        out = self.getOut()

        mask = moment_params.mask
        ys = np.where(mask, moment_params.ys, 0)

        if (ys.shape[0] > 1) and np.allclose(ys[1:] - ys[:1], 0):
            mask = mask[:1]
            ys = ys[:1]

        dys = np.where(mask, moment_params.dys, 0)
        _out = skewness(
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

        return out.freeze()

    @lru_cache
    def _measureKurtosis(
        self,
        *,
        from_model: bool = True,
        excess: bool = False,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Calculates the kurtosis of all samples.
        """
        if bg_flux is None:
            return self._measureKurtosis(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )

        if excess:
            return self._measureKurtosis(
                from_model=from_model,
                bg_flux=bg_flux,
            ) - 3.0

        moment_params = self._getMomentParams(
            from_model=from_model,
            bg_flux=bg_flux,
            centre="wave",
        )
        if from_model:
            return moment_params.cm4 / moment_params.cm2**2

        from ..spectrum_inner import kurtosis

        out = self.getOut()

        mask = moment_params.mask
        ys = np.where(mask, moment_params.ys, 0)

        if (ys.shape[0] > 1) and np.allclose(ys[1:] - ys[:1], 0):
            mask = mask[:1]
            ys = ys[:1]

        dys = np.where(mask, moment_params.dys, 0)
        _out = kurtosis(
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

        return out.freeze()
    

    def _measureStrength(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        """
        Proxy for `measureZerothMoment`.
        """
        if bg_flux is None:
            return self._measureStrength(
                from_model=from_model,
                bg_flux=BackgroundFlux({"all", "em"}),
            )
        
        return self._measureZerothMoment(
            from_model=from_model,
            bg_flux=bg_flux
        )

    ### Public methods

    @wraps(_measureZerothMoment)
    @pydantic.validate_call
    def measureZerothMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureZerothMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureFirstMoment)
    @pydantic.validate_call
    def measureFirstMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureFirstMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureSecondCentralMoment)
    @pydantic.validate_call
    def measureSecondCentralMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureSecondCentralMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureThirdCentralMoment)
    @pydantic.validate_call
    def measureThirdCentralMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureThirdCentralMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureFourthCentralMoment)
    @pydantic.validate_call
    def measureFourthCentralMoment(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureFourthCentralMoment(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureCentroid)
    @pydantic.validate_call
    def measureCentroid(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureCentroid(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureCentroidOffset)
    @pydantic.validate_call
    def measureCentroidOffset(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureCentroidOffset(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureDispersion)
    @pydantic.validate_call
    def measureDispersion(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureDispersion(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureVelocityDispersion)
    @pydantic.validate_call
    def measureVelocityDispersion(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureVelocityDispersion(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureSkewness)
    @pydantic.validate_call
    def measureSkewness(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureSkewness(
            from_model=from_model,
            bg_flux=bg_flux,
        )

    @wraps(_measureKurtosis)
    @pydantic.validate_call
    def measureKurtosis(
        self,
        *,
        from_model: bool = True,
        excess: bool = False,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureKurtosis(
            from_model=from_model,
            excess=excess,
            bg_flux=bg_flux,
        )

    @wraps(_measureStrength)
    @pydantic.validate_call
    def measureStrength(
        self,
        *,
        from_model: bool = True,
        bg_flux: BackgroundFlux | None = None,
    ) -> FrozenMeasureResult:
        return self._measureStrength(
            from_model=from_model,
            bg_flux=bg_flux,
        )

