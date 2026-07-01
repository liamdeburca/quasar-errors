from typing import Self, Literal, ClassVar
from numpy import array, zeros_like, float64

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from quasar_core.modelling.host import HostGalaxyEvaluate, evaluate_exact

from quasar_models.host import (
    HostGalaxyModel, 
    HostGalaxyTemplate,
)
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_

from quasar_utils.decorators import validate_call

from ..utils.base_sample import BaseSample

@dataclass(frozen=True, kw_only=True)
class HostGalaxySample(BaseSample):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ignored_types=(HostGalaxyEvaluate,),
    )

    flux: float
    fwhm: float

    name: str = ""
    template: HostGalaxyTemplate

    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None = None    
    evaluate_func: HostGalaxyEvaluate = evaluate_exact

    model_type: ClassVar[Literal["hg"]] = "hg"

    @classmethod
    @validate_call
    def fromHostGalaxyModel(cls, model: HostGalaxyModel) -> Self:
        return HostGalaxySample(
            flux=model.flux.value,
            fwhm=model.fwhm.value,
            name=model.name,
            template=model.template,
            **model._interpolation_matrices,
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
            self.flux, self.fwhm, 
            template_fwhm=self.template.fwhm,
            template_x=self.template.x,
            template_data=self.template.data / self.template.normalisation,
            sigma_res=self.template.sigma_res,
            n_scales=self.template.n_scales,
            interpolation_matrix=self.interpolation_matrix,
            y=y,
        )
    
    @property
    def sorting_key(self) -> tuple[float, float]: 
        return (3.0, 0.0)