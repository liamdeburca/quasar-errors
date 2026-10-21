from collections.abc import Callable, Iterator
from typing import Self, TypedDict

import pydantic
from quasar_models.modeling import SequentialModel
from quasar_typing.numpy import FloatVector

from ..model_samples import Sample


class _SerializationDict(TypedDict):
    n_samples: int
    n_pix: int
    x: FloatVector
    dy: FloatVector
    dx: FloatVector
    ys: list[FloatVector]
    weights: tuple[float, ...]
    seq_model: SequentialModel | None
    parameters: list[FloatVector | None]


class ErrorResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    seq_model: SequentialModel | None
    samples: tuple[Sample, ...] = pydantic.Field(default_factory=tuple)
    sample_weights: tuple[float, ...] = pydantic.Field(default_factory=tuple)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_pix(self) -> int:
        return self.samples[0].x.size

    ### Coordinates

    @property
    def x(self) -> FloatVector:
        return self.samples[0].x

    @property
    def y(self) -> FloatVector:
        return self.samples[0].y

    @property
    def dy(self) -> FloatVector:
        return self.samples[0].dy

    @property
    def dx(self) -> FloatVector:
        return self.samples[0].dx

    ###

    def __repr__(self) -> str:
        _result = hex(id(self))
        return f"<ErrorResult {_result} w/ {self.n_samples} samples>"

    def __getitem__(self, index: int | slice) -> Sample | tuple[Sample]:
        if isinstance(index, slice):
            return tuple(self.samples[index])
        return self.samples[index]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return self.n_samples

    @pydantic.validate_call
    def sample_filter(
        self,
        *,
        target_pattern: str | tuple[str, ...] | Callable[[str], bool] | None = None,
        target_wave: float | set[float] | None = None,
        target_line: str | set[str] | None = None,
    ) -> Callable[[Sample], Sample]:
        """
        Returns a function that filters samples based on the target pattern, 
        line, or wave.
        """
        from numpy import zeros_like as backup

        base_sample = self.samples[0]
        if target_pattern is not None:
            method_name = "splitAtPattern"
            arg = target_pattern
        elif target_wave is not None:
            method_name = "splitAtWave"
            arg = target_wave
        elif target_line is not None:
            method_name = "splitAtLine"
            arg = target_line
        else:
            raise ValueError("Must provide either pattern, wave, or line.")

        extra = base_sample._split(method_name, arg)[1]
        if (extra is not None) and (extra.em is not None):
            backup = extra.em

        def sample_filter(sample: Sample, _backup=backup) -> Sample:
            return sample._split(method_name, arg, backup=_backup)[0]

        return sample_filter

    @pydantic.validate_call
    def isolate(
        self,
        *,
        target_pattern: str | tuple[str, ...] | Callable[[str], bool] | None = None,
        target_wave: float | set[float] | None = None,
        target_line: str | set[str] | None = None,
    ) -> Self:
        from quasar_models.modeling import PrepareModel

        sample_filter = self.sample_filter(
            target_pattern=target_pattern,
            target_line=target_line,
            target_wave=target_wave,
        )
        samples = tuple(map(sample_filter, self.samples))

        seq_model = None
        if self.seq_model is not None:
            chosen_submodels = sorted(
                [
                    self.seq_model._model[name].copy()
                    for name in samples[0].names
                ],
                key=lambda m: m.sorting_key,
            )
            if chosen_submodels:
                preparer = PrepareModel(
                    x=self.x,
                    model=sum(chosen_submodels[1:], start=chosen_submodels[0]),
                    kernels=self.seq_model.kernels,
                )
                with preparer as (seq_model, _):
                    pass
            
        return ErrorResult(
            seq_model=seq_model,
            samples=samples,
            sample_weights=self.sample_weights,
        )

    def _serialization_helper(self) -> _SerializationDict:
        """
        Helper function for serializing the ErrorResult object.
        """
        return {
            "n_samples": self.n_samples,
            "n_pix": self.n_pix,
            "x": self.x,
            "dy": self.dy,
            "dx": self.dx,
            "ys": [s.y for s in self.samples],
            "weights": self.sample_weights,
            "seq_model": self.seq_model,
            "parameters": [s.parameters for s in self.samples],
        }