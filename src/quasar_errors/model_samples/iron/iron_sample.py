from typing import Self, Literal
from pydantic import validate_call
from pydantic.dataclasses import dataclass

from quasar_models.iron import IronModel, IronTemplate, evaluation
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

@dataclass(frozen=True, kw_only=True)
class IronSample:
    """
    ** PYDANTIC DATACLASS **
    """
    flux: float
    fwhm: float
    split: float
    left: float
    right: float

    template: IronTemplate

    name: str = ""
    perform_interp_fitting: bool = False

    @classmethod
    @validate_call
    def fromIronModel(
        cls,
        model: IronModel,
    ) -> Self:
        return IronSample(
            flux=model.flux.value,
            fwhm=model.fwhm.value,
            split=model.split.value,
            left=model.left.value,
            right=model.right.value,
            template=model.template,
            name=model.name,
            perform_interp_fitting=model._perform_interp_fitting,
        )
    
    def __call__(
            self, 
            x: float | FloatVector,
            interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None,
        ) -> float | FloatVector:
        if self.perform_interp_fitting:
            return evaluation.evaluate_interp(
                x, self.flux, self.fwhm,
                template=self.template,
                interpolation_matrix=interpolation_matrix,
            )
        
        return evaluation.evaluate(
            x,
            self.flux, self.fwhm, self.split, self.left, self.right,
            template=self.template,
            interpolation_matrix=interpolation_matrix,
        )
        
    @property
    def model_type(self) -> Literal["fe"]: 
        return "fe"

    @property
    def sorting_key(self) -> tuple[float, float]: 
        return (1.0, self.template.x[0])