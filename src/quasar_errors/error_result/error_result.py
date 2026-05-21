from typing import Self, Callable, Iterator
from numpy import zeros_like
from dataclasses import field
from functools import cached_property

from pydantic import validate_call
from pydantic.dataclasses import dataclass

from ..model_samples import Sample

from quasar_typing.numpy import FloatVector

@dataclass(frozen=True, repr=False)
class ErrorResult:
    samples: tuple[Sample,...] = field(default_factory=tuple)
    sample_weights: tuple[float,...] = field(default_factory=tuple)

    @cached_property
    def n_samples(self) -> int: return len(self.samples)

    @cached_property
    def n_pix(self) -> int: return self.samples[0].x.size

    @cached_property
    def x(self) -> FloatVector: return self.samples[0].x

    @cached_property
    def y(self) -> FloatVector: return self.samples[0].y

    @cached_property
    def dy(self) -> FloatVector: return self.samples[0].dy

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
    
    @validate_call
    def sample_filter(
        self,
        target_line: str | set[str] | None = None,
        target_wave: float | set[float] | None = None,
    ) -> Callable[[Sample], Sample]:
        """
        ** PYDANTIC VALIDATED METHOD **

        Returns a function that filters samples based on the target line or 
        wave.
        """
        assert target_line or target_wave
        assert not (target_line and target_wave)

        backup = zeros_like        
        base_sample = self.samples[0]
        if target_wave is not None:
            _, extra = base_sample.splitAtWave.__wrapped__(
                base_sample,
                target_wave,
            )
            if (extra is not None) and (extra.em is not None):
                backup = extra.em

            def sample_filter(sample: Sample) -> Sample:
                nonlocal target_wave, backup
                return sample.splitAtWave.__wrapped__(
                    sample,
                    target_wave, 
                    backup=backup,
                )[0]
        else:
            _, extra = base_sample.splitAtLine.__wrapped__(
                base_sample,
                target_line,
            )
            if (extra is not None) and (extra.em is not None):
                backup = extra.em

            def sample_filter(sample: Sample) -> Sample:
                nonlocal target_line, backup
                return sample.splitAtLine.__wrapped__(
                    sample,
                    target_line, 
                    backup=backup,
                )[0]

        return sample_filter
    
    @validate_call
    def isolate(
        self,
        target_line: str | set[str] | None = None,
        target_wave: float | set[float] | None = None,
    ) -> Self:
        """
        ** PYDANTIC VALIDATED METHOD **
        """
        sample_filter = self.sample_filter.__wrapped__(
            self,
            target_line=target_line,
            target_wave=target_wave,
        )
        return self.__class__(
            samples=tuple(map(sample_filter, self.samples)),
            sample_weights=self.sample_weights,
        )