"""
This submodule contains dataclasses designed to represent submodels used to 
fit quasar spectra. They act as simpler, imutable versions of AstroPy models, 
but with less functionality and weight, making them suitable for error 
estimation where many samples are generated.
"""
__all__ = [
    "Sample",
    "PowerLawSample", 
    "IronSample", "IronSampleList",
    "BalmerSample",
    "GaussianSample", "GaussianSampleList",
]
from .sample import Sample
from .continuum import PowerLawSample
from .iron import IronSample, IronSampleList
from .balmer import BalmerSample
from .line import GaussianSample, GaussianSampleList