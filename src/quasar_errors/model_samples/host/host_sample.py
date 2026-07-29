from dataclasses import dataclass
from typing import ClassVar, Literal, Self

from numpy import array, float64, zeros_like
from quasar_models._core.modeling.host import (
    HostGalaxyEvaluate,
    evaluate_exact,
)
from quasar_models._core.modeling.template.cytemplate import CyTemplate
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
    flux: float
    fwhm: float

    name: str
    template: HostGalaxyTemplate
    cytemplate: CyTemplate | None
    n_scales: float
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None
    
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
            cytemplate=model.cytemplates['cytemplate'],
            n_scales=model.n_scales,
            interpolation_matrix=model.interpolation_matrix,
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
            template=self.template,
            cytemplate=self.cytemplate,
            n_scales=self.n_scales,
            interpolation_matrix=self.interpolation_matrix,
            y=y,
        )

    @property
    def sorting_key(self) -> tuple[float, float]:
        return (3.0, 0.0)
