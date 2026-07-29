"""
This submodule contains dataclasses designed to represent submodels used to
fit quasar spectra. They act as simpler, imutable versions of AstroPy models,
but with less functionality and weight, making them suitable for error
estimation where many samples are generated.
"""

__all__ = [
    "BalmerSample",
    "GaussianSample",
    "GaussianSampleList",
    "HostGalaxySample",
    "IronSample",
    "IronSampleList",
    "PowerLawSample",
    "Sample",
]
from .balmer import BalmerSample
from .continuum import PowerLawSample
from .host import HostGalaxySample
from .iron import IronSample, IronSampleList
from .line import GaussianSample, GaussianSampleList
from .sample import Sample
