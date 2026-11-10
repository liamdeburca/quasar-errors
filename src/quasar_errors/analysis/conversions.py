"""
Set of conversions between rest-frame and observed-frame quantities.

Please note, the wavelength grid was already converted to rest-frame (and the 
flux densities accordingly) during loading, so conversions in this file assume 
that a `1 + z` correction has already been applied.
"""
import math
from typing import Any


class WavelengthConversions:
    """
    Conversion between rest-frame and observed-frame wavelengths.
    """
    @classmethod
    def rest_to_observed[V](cls, value: V, z: float) -> V:
        return value * (1.0 + z)

    @classmethod
    def observed_to_rest[V](cls, value: V, z: float) -> V:
        return value / (1.0 + z)


class FluxDensityConversions:
    @classmethod
    def rest_to_observed[V](cls, value: V, z: float) -> V:
        return value / (1.0 + z)

    @classmethod
    def observed_to_rest[V](cls, value: V, z: float) -> V:
        return value * (1.0 + z)

    @classmethod
    def flux_density_to_luminosity_density[V](cls, value: V, d_l: Any) -> V:
        return 4 * math.pi * d_l**2 * value

    @classmethod
    def luminosity_density_to_flux_density[V](cls, value: V, d_l: Any) -> V:
        return value / (4 * math.pi * d_l**2)

    @classmethod
    def rest_to_observed_luminosity[V](cls, value: V, z: float, d_l: Any) -> V:
        return cls.flux_density_to_luminosity_density(
            cls.rest_to_observed(value, z), 
            d_l,
        )

    @classmethod
    def observed_to_rest_luminosity[V](cls, value: V, z: float, d_l: Any) -> V:
        return cls.luminosity_density_to_flux_density(
            cls.observed_to_rest(value, z),
            d_l,
        )