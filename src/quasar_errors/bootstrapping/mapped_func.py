from dataclasses import field
from typing import Self

from pydantic.dataclasses import dataclass
from quasar_models.modeling import Fitter, PrepareModel
from quasar_typing.astropy import Model_
from quasar_typing.numpy import FittableFloatVector
from quasar_utils.setup import FitterKwargs

from quasar_errors.model_samples import Sample


@dataclass
class MappedFunc:
    x: FittableFloatVector
    dy: FittableFloatVector
    y_bg: FittableFloatVector

    variant: str = field(default="spectrum", kw_only=True)
    model: Model_ | None = field(default=None, kw_only=True)
    fitter_kwargs: FitterKwargs | None = field(default=None, kw_only=True)

    model_is_prepared: bool = field(default=False, kw_only=True)
    weights: FittableFloatVector | None = field(default=None, kw_only=True)
    fitter: Fitter | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if not self.x.size == self.dy.size == self.y_bg.size:
            msg = f"x ({self.x.size}), dy ({self.dy.size}), and y_bg ({self.y_bg.size}) must have the same size"
            raise ValueError(msg)

        if self.variant != "spectrum":
            if self.model is None:
                msg = f"'model' must be provided for variant '{self.variant}'"
                raise ValueError(msg)

            if self.fitter is None:
                msg = f"'fitter' must be provided for variant '{self.variant}'"
                raise ValueError(msg)

    def __enter__(self) -> Self:
        if self.variant != "spectrum":
            PrepareModel.prepare_model(self.x, self.model)
        self.model_is_prepared = True
        self.weights = 1.0 / self.dy
        self.fitter = Fitter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.variant != "spectrum":
            PrepareModel.unprepare_model(self.model)
        self.model_is_prepared = False
        self.weights = None
        self.fitter = None

    def __call__(
        self,
        y: FittableFloatVector,
    ) -> Sample:
        if not self.model_is_prepared:
            msg = "Instances of 'MappedFunc' must be used as context managers."
            raise RuntimeError(msg)

        _y = y - self.y_bg
        if self.variant == "spectrum":
            return Sample.fromSpectrumSample(
                self.x,
                _y,
                self.dy,
                weight=1.0,
            )

        self.fitter(
            self.model,
            self.x,
            _y,
            weights=self.weights,
            **self.fitter_kwargs,
        )
        return Sample.fromParamVector(
            self.x,
            _y,
            self.dy,
            self.fitter.sol,
            self.model,
            weight=1.0,
        )

    def __getstate__(self) -> dict:
        return {
            "x": self.x,
            "dy": self.dy,
            "y_bg": self.y_bg,
            "variant": self.variant,
            "model": self.model,
            "fitter_kwargs": self.fitter_kwargs,
            "model_is_prepared": self.model_is_prepared,
            "weights": self.weights,
            "fitter": self.fitter,
        }

    def __setstate__(self, state: dict) -> None:
        self.x = state["x"]
        self.dy = state["dy"]
        self.y_bg = state["y_bg"]
        self.variant = state["variant"]
        self.model = state["model"]
        self.fitter_kwargs = state["fitter_kwargs"]
        self.model_is_prepared = state["model_is_prepared"]
        self.weights = state["weights"]
        self.fitter = state["fitter"]
