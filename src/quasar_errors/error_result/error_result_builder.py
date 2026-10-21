from collections.abc import Iterable, Iterator
from typing import Self

import pydantic
from quasar_models.modeling import SequentialModel
from quasar_typing.numpy import CoordsTuple, FloatVector

from ..model_samples import Sample
from .error_result import ErrorResult


class ErrorResultBuilder(pydantic.BaseModel):
    seq_model: SequentialModel | None
    samples: list[Sample] = pydantic.Field(default_factory=list)
    sample_weights: list[float] = pydantic.Field(default_factory=list)

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

    @classmethod
    def initialise(
        cls,
        coords: CoordsTuple,
        seq_model: SequentialModel,
    ) -> Self:
        sample = Sample.fromModelSample(
            *coords,
            model=seq_model,
            weight=1.0,
        )
        return ErrorResultBuilder(
            seq_model=seq_model,
            samples=[sample],
            sample_weights=[1.0],
        )

    @pydantic.validate_call
    def simple_append(self, sample: Sample) -> None:
        self.samples.append(sample)
        self.sample_weights.append(1.0)

    @pydantic.validate_call
    def simple_extend(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self.simple_append.__wrapped__(self, sample)

    def finalise(self) -> ErrorResult:
        return ErrorResult(
            seq_model=self.seq_model,
            samples=tuple(self.samples),
            sample_weights=tuple(self.sample_weights),
        )
