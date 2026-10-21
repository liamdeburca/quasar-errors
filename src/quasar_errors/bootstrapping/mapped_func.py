from dataclasses import field
from typing import Optional, Self

import numpy as np
import pydantic
from quasar_models.modeling import Fitter, PrepareModel, SequentialModel
from quasar_typing.astropy import ModelType
from quasar_typing.numpy import (
    BoolVector,
    FittableFloatVector,
    FloatMatrix,
    FloatVector,
)
from quasar_utils.setup import FitterKwargs

from quasar_errors.model_samples import Sample


@pydantic.dataclasses.dataclass
class MappedFunc:
    x: FittableFloatVector
    dy: FloatVector
    dx: FloatVector
    y_bg: FloatVector

    variant: str = field(default="spectrum", kw_only=True)
    model: Optional[ModelType] = field(default=None, kw_only=True)
    is_logbinned: bool | None = field(default=None, kw_only=True)
    
    eval_mask: BoolVector | None = field(default=None, kw_only=True)
    comp_mask: BoolVector | None = field(default=None, kw_only=True)
    kernels: FloatMatrix | None = field(default=None, kw_only=True)

    fitter_kwargs: FitterKwargs | None = field(default=None, kw_only=True)

    _preparer: PrepareModel | None = field(default=None, kw_only=True)
    _seq_model: SequentialModel | None = field(default=None, kw_only=True)
    _where: BoolVector | None = field(default=None, kw_only=True)
    _model_is_prepared: bool = field(default=False, kw_only=True)
    _weights: FittableFloatVector | None = field(default=None, kw_only=True)
    _fitter: Fitter | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.x.ndim != 1:
            msg = f"`x` must be a 1-dimensional array, but got shape {self.x.shape}"
            raise ValueError(msg)
        
        if self.x.shape != self.dy.shape:
            msg = f"Shape mismatch between `x` ({self.x.shape}) and `dy` ({self.dy.shape})"
            raise ValueError(msg)
        if self.x.shape != self.dx.shape:
            msg = f"Shape mismatch between `x` ({self.x.shape}) and `dx` ({self.dx.shape})"
            raise ValueError(msg)
        if self.x.shape != self.y_bg.shape:
            msg = f"Shape mismatch between `x` ({self.x.shape}) and `y_bg` ({self.y_bg.shape})"
            raise ValueError(msg)
    
        if self.variant != "spectrum" and self.model is None:
            msg = f"'model' must be provided for variant '{self.variant}'"
            raise ValueError(msg)

        if self.eval_mask is None:
            self.eval_mask = np.ones(self.x.size, np.bool_)
        elif self.eval_mask.shape != self.x.shape:
            msg = f"Shape mismatch between `eval_mask` ({self.eval_mask.shape}) and `x` ({self.x.shape})"
            raise ValueError(msg)
        
        if self.comp_mask is None:
            self.comp_mask = np.ones(self.x.size, np.bool_)
        elif self.comp_mask.shape != self.x.shape:
            msg = f"Shape mismatch between `comp_mask` ({self.comp_mask.shape}) and `x` ({self.x.shape})"
            raise ValueError(msg)
        
        if self.kernels is None:
            self.kernels = np.ones((self.x.size, 1), np.float64)
        elif self.kernels.shape[0] != self.x.size:
            msg = f"Dimension 0 mismatch between `kernels` ({self.kernels.shape}) and `x` ({self.x.shape})"
            raise ValueError(msg)

    def __enter__(self) -> Self:
        if self.variant != "spectrum":
            self._preparer = PrepareModel(
                x=self.x,
                model=self.model,
                eval_mask=self.eval_mask,
                comp_mask=self.comp_mask,
                kernels=self.kernels,
                is_logbinned=self.is_logbinned,
            )
            self._seq_model, self._where = self._preparer.__enter__()

        self._model_is_prepared = True
        self._weights = 1.0 / self.dy
        self._fitter = Fitter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.variant != "spectrum":
            self._preparer.__exit__(None, None, None)
            self._seq_model = None
            self._where = None

        self._model_is_prepared = False
        self._weights = None
        self._fitter = None

    def __call__(self, y: FittableFloatVector) -> Sample:
        if not self._model_is_prepared:
            msg = "Instances of 'MappedFunc' must be used as context managers."
            raise RuntimeError(msg)

        mask = self.comp_mask & self.eval_mask
        _y = y - self.y_bg
        args = (self.x[mask], _y[mask], self.dy[mask], self.dx[mask])
        
        if self.variant == "spectrum":
            return Sample.fromSpectrumSample.__wrapped__(
                Sample,
                *args,
                weight=1.0,
            )
        
        self._fitter(
            self._seq_model, 
            self.x, 
            _y,
            weights=self._weights,
            where=self._where,
            **(self.fitter_kwargs or {}),
        )
        return Sample.fromParamVector.__wrapped__(
            Sample,
            *args,
            self._fitter.sol, 
            self._seq_model,
            weight=1.0,
        )
    