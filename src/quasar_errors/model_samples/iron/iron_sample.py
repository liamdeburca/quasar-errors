from typing import Self, Literal, ClassVar
from numpy import array, zeros_like, float64
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from quasar_core.modelling.iron import IronEvaluate, evaluate_exact

from quasar_models.iron import IronModel, IronTemplate
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

from quasar_utils.decorators import validate_call

from ..utils.base_sample import BaseSample

@dataclass(frozen=True, kw_only=True)
class IronSample(BaseSample):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(IronEvaluate,),
    )

    flux: float
    fwhm: float
    split: float
    left: float
    right: float

    name: str = ""
    template: IronTemplate
    
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None
    evaluate_func: IronEvaluate = evaluate_exact

    model_type: ClassVar[Literal["fe"]] = "fe"

    @classmethod
    @validate_call
    def fromIronModel(cls, model: IronModel) -> Self:
        return IronSample(
            flux=model.flux.value,
            fwhm=model.fwhm.value,
            split=model.split.value,
            left=model.left.value,
            right=model.right.value,
            name=model.name,
            template=model.template,
            **model._interpolation_matrices, # Interpolation matrix
            evaluate_func=model.evaluate_func,
        )
    
    def __call__(
        self, 
        x: float | FloatVector, 
        y: FloatVector | None = None,
    ) -> float | FloatVector:
        if isinstance(x, (int, float)):
            assert y is None
            _x = array([x], dtype=float64)
            return self(_x)[0]
        
        if y is None:
            y = zeros_like(x, dtype=float64)
        return self.evaluate_func(
            x,
            self.flux, self.fwhm, self.split, self.left, self.right,
            template_fwhm=self.template.fwhm,
            template_x=self.template.x,
            template_data=self.template.data / self.template.normalisation,
            sigma_res=self.template.sigma_res,
            scale=self.template.scale,
            n_scales=self.template.n_scales,
            interpolation_matrix=self.interpolation_matrix,
            y=y,
        )
        
    @property
    def sorting_key(self) -> tuple[float, float]: 
        return (1.0, self.template.x[0])