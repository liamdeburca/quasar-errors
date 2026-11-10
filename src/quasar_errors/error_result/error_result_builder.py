from collections.abc import Iterable, Iterator
from typing import Any, Self

from pydantic import BaseModel, Field, validate_call
from quasar_models.modeling import SequentialModel
from quasar_typing.numpy import CoordsTuple, FloatVector

from ..model_samples import Sample
from .error_result import ErrorResult


class ErrorResultBuilder(BaseModel):
    seq_model: SequentialModel | None
    samples: list[Sample] = Field(default_factory=list)
    sample_weights: list[float] = Field(default_factory=list)

    def __getitem__(self, index: int | slice) -> Sample | list[Sample]:
        return self.samples[index]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_pix(self) -> int:
        return self.samples[0].x.size

    @property
    def x(self) -> FloatVector:
        return self.samples[0].x

    @property
    def y(self) -> FloatVector:
        return self.samples[0].y

    @property
    def dy(self) -> FloatVector:
        return self.samples[0].dy

    ### initialise

    @classmethod
    def _initialise(
        cls,
        coords: CoordsTuple,
        seq_model: SequentialModel,
    ) -> dict[str, Any]:
        sample = Sample._fromModelSample(
            *coords,
            model=seq_model,
            weight=1.0,
        )
        return {
            "seq_model": seq_model,
            "samples": [sample],
            "sample_weights": [1.0],
        }

    @classmethod
    @validate_call
    def initialise(
        cls,
        coords: CoordsTuple,
        seq_model: SequentialModel,
    ) -> Self:
        return cls(**cls._initialise(coords, seq_model))

    ### simple_append

    def _simple_append(self, sample: Sample) -> None:
        self.samples.append(sample)
        self.sample_weights.append(1.0)

    @validate_call
    def simple_append(self, sample: Sample) -> None:
        self._simple_append(sample)

    ### simple_extend

    def _simple_extend(self, samples: Iterable[Sample]) -> None:
        [self._simple_append(sample) for sample in samples]

    @validate_call
    def simple_extend(self, samples: Iterable[Sample]) -> None:
        return self._simple_extend(samples)

    ### finalise

    def _finalise(self) -> dict[str, Any]:
        return {
            "seq_model": self.seq_model,
            "samples": tuple(self.samples),
            "sample_weights": tuple(self.sample_weights),
        }

    def finalise(self) -> ErrorResult:
        return ErrorResult(**self._finalise())