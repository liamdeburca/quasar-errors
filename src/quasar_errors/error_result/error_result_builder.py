from collections.abc import Iterable, Iterator
from dataclasses import field
from typing import Self

from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Model_
from quasar_typing.numpy import CoordsTuple, FloatVector
from quasar_utils.decorators import validate_call

from ..model_samples import Sample
from .error_result import ErrorResult


@dataclass
class ErrorResultBuilder:
    samples: list[Sample] = field(default_factory=list)
    sample_weights: list[float] = field(default_factory=list)

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
        model: Model_,
    ) -> Self:
        sample = Sample.fromModelSample(
            *coords,
            model=model,
            weight=1.0,
        )
        return ErrorResultBuilder(
            samples=[sample],
            sample_weights=[1.0],
        )

    @validate_call
    def simple_append(self, sample: Sample) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        self.samples.append(sample)
        self.sample_weights.append(1.0)

    @validate_call
    def simple_extend(self, samples: Iterable[Sample]) -> None:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        for sample in samples:
            self.simple_append.__wrapped__(self, sample)

    def finalise(self) -> ErrorResult:
        return ErrorResult(
            samples=tuple(self.samples),
            sample_weights=tuple(self.sample_weights),
        )
