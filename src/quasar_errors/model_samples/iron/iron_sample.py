from dataclasses import dataclass
from typing import ClassVar, Literal, Self

from numpy import array, float64, zeros_like
from quasar_models._core.modeling.iron import IronEvaluate, evaluate_exact
from quasar_models._core.modeling.template.cytemplate import CyTemplate
from quasar_models.iron import IronModel, IronTemplate
from quasar_typing.numpy import FloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_utils.decorators import validate_call

from ..utils.base_sample import BaseSample


@dataclass(frozen=True, kw_only=True)
class IronSample(BaseSample):
    flux: float
    fwhm: float
    split: float
    left: float
    right: float

    name: str
    template: IronTemplate
    cytemplate: CyTemplate | None
    scale: float
    n_scales: float
    interpolation_matrix: tuple[csr_matrix_, FloatVector] | None
    
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
            cytemplate=model.cytemplates['cytemplate'],
            scale=model.scale,
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
            self.flux, self.fwhm, self.split, self.left, self.right,
            template=self.template,
            cytemplate=self.cytemplate,
            scale=self.scale,
            n_scales=self.n_scales,
            interpolation_matrix=self.interpolation_matrix,
            y=y,
        )

    @property
    def sorting_key(self) -> tuple[float, float]:
        return (1.0, self.template.x[0])
