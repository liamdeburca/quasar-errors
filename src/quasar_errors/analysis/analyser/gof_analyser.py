import math

import pydantic
from quasar_typing.numpy import FloatVector
from quasar_typing.string_selections import BackgroundFlux

from ..measure_result import FrozenMeasureResult
from .base_analyser import _BaseAnalyser


@pydantic.dataclasses.dataclass
class GoFAnalyser(_BaseAnalyser):
    """
    GoFAnalyser - Goodness of Fit Analyser

    This class provides methods to analyse the goodness of fit of the best-fit
    model to the observed data.

    Notes
    -----
    Private methods are prefixed with an underscore (_) and are intended for 
    internal use within the class, or use by an experienced developer. These 
    methods are sometimes cached (LRU). 

    External methods have no underscore prefix and are intended for the general 
    user. They are simple wrappers of their corresponding private methods using 
    `pydantic.validate_call`.
    """
    def _get_residuals(self) -> FloatVector:
        """
        Compute the masked vector of residuals for the best-fit model 
        (first sample).
        """
        z = self._getFlux(
            bg_flux=BackgroundFlux({"all"}), # Subtract all model components
            ensure_not_negative=False,
            simplify=False,
        )[0] / self.dy # Normalise by the observational uncertainties
        mask = self._getMask(
            centre="wave",
            from_model=True,
            simplify=False,
        )[0]
        return z[mask]

    def _get_em_n_free(self) -> int:
        """
        Naively compute the number of free parameters in the emission line 
        model(s).

        This method assumes the following:
        - `GaussianSample` instances with `is_v_profile_copy = True` have 1 free 
        parameter (`strength_scale`).
        - `GaussianSample` instances with `is_v_profile_copy = False` have 3 
        free parameters (`strength`, `fwhm_v`, `v_off`).

        No other model components are considered.
        """
        em_samples = self.error_result[0].em
        if em_samples is None:
            return 0
        
        return sum(
            1 if em_sample.is_v_profile_copy else 3
            for em_sample in em_samples
        )

    ### Public Methods

    def measureReducedChi2(self) -> FrozenMeasureResult:
        """
        Calculate the reduced chi-squared of the residuals. 
        
        Notes
        -----
        The number of degrees of freedom is calculated naively -- by counting 
        only Gaussian emission line models and assuming velocity-profile copies
        have just 1 free parameter, while other Gaussian emission line models 
        have 3.
        """
        out = self.getOut()
        z = self._get_residuals()
        n = z.size
        n_free = self._get_em_n_free()
        
        if n <= n_free:
            stat = math.inf
        else:
            stat = (z**2).sum() / (n - n_free)
        
        out.val[0] = stat
        out.unc[0] = 0.0
        
        return out.freeze()

    def measureChi2Significance(self) -> FrozenMeasureResult:
        """
        Calculate the significance of the chi-squared of the residuals.
        """
        from scipy.stats import chi2

        out = self.getOut()
        z = self._get_residuals()
        
        out.val[0] = chi2.sf((z**2).mean(), 1)
        out.unc[0] = 0.0
        
        return out.freeze()

    def measureKS(self) -> FrozenMeasureResult:
        """
        Calculate the Kolmogorov-Smirnov (KS) statistic of the residuals.
        """
        from scipy.stats import kstest

        out = self.getOut()
        z = self._get_residuals()
        
        out.val[0] = kstest(z, "norm").statistic
        out.unc[0] = 0.0
        
        return out.freeze()

    def measureKSSignificance(self) -> FrozenMeasureResult:
        """
        Calculate the significance of the Kolmogorov-Smirnov (KS) statistic of 
        the residuals.
        """
        from scipy.stats import kstest

        out = self.getOut()
        z = self._get_residuals()

        out.val[0] = kstest(z, "norm").pvalue
        out.unc[0] = 0.0

        return out.freeze()

    def measureFFT(self) -> FrozenMeasureResult:
        """
        Calculate the FFT test statistic on the residuals.
        """
        from quasar_utils.absorption.absorption import _fft_test

        out = self.getOut()
        z = self._get_residuals()
        stat = _fft_test(z)[0]

        out.val[0] = stat
        out.unc[0] = 0.0

        return out.freeze()

    def measureFFTSignificance(self) -> FrozenMeasureResult:
        """
        Calculate the significance of the FFT test statistic on the residuals.
        """
        from quasar_utils.absorption.absorption import _fft_test

        out = self.getOut()
        z = self._get_residuals()
        
        out.val[0] = _fft_test(z)[1]
        out.unc[0] = 0.0

        return out.freeze()