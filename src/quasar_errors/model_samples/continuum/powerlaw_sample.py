from typing import Self, Literal
from functools import cached_property
from pydantic import validate_call
from pydantic.dataclasses import dataclass

from quasar_models.continuum import PowerLawModel, evaluation
from quasar_typing.numpy import FloatVector

@dataclass(frozen=True)
class PowerLawSample:
    """
    ** PYDANTIC DATACLASS **
    """
    flux: float
    alpha: float
    x0: float
    name: str = ""
    
    @classmethod
    @validate_call
    def fromPowerLawModel(
        cls,
        model: PowerLawModel,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        return PowerLawSample(
            model.flux.value,
            model.alpha.value,
            model.x0,
            name=model.name,
        )
    
    def __call__(self, x: float | FloatVector) -> float | FloatVector:
        return evaluation.evaluate(x, self.flux, self.alpha, self.x0)
    
    @cached_property
    def model_type(self) -> Literal["pl"]: 
        return "pl"

    @cached_property
    def sorting_key(self) -> tuple[float, float]: 
        return (0.0, self.x0)