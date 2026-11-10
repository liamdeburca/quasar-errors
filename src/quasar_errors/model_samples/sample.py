from collections import defaultdict
from collections.abc import Callable
from typing import Any, Self

from numpy import concatenate, zeros_like
from pydantic import BaseModel, validate_call
from quasar_models.modeling import SequentialModel
from quasar_typing.numpy import FloatVector
from quasar_typing.re import StrPattern

from .balmer.balmer_sample import BalmerSample
from .continuum.powerlaw_sample import PowerLawSample
from .host.host_sample import HostGalaxySample
from .iron.iron_sample import IronSample
from .iron.iron_sample_list import IronSampleList
from .line.gaussian_sample import GaussianSample
from .line.gaussian_sample_list import GaussianSampleList
from .utils.base_sample import BaseSample


class Sample(BaseModel, frozen=True):
    x: FloatVector
    y: FloatVector
    dy: FloatVector
    dx: FloatVector

    pl: PowerLawSample | None = None
    fe: IronSampleList | None = None
    ba: BalmerSample | None = None
    hg: HostGalaxySample | None = None
    em: GaussianSampleList | None = None

    weight: float = 1.0

    def __str__(self) -> str:
        return f"Sample at <{hex(id(self))}>"

    def __repr__(self) -> str:
        return "\n".join(
            [
                self.__str__(),
                f"   x=<{hex(id(self.x))}>",
                f"   y=<{hex(id(self.y))}>",
                f"  dy=<{hex(id(self.dy))}>",
                f"  dx=<{hex(id(self.dx))}>",
                f"  pl=<{'/' if self.pl is None else hex(id(self.pl))}>",
                f"  fe=<{'/' if self.fe is None else hex(id(self.fe))}>",
                f"  ba=<{'/' if self.ba is None else hex(id(self.ba))}>",
                f"  hg=<{'/' if self.hg is None else hex(id(self.hg))}>",
                f"  em=<{'/' if self.em is None else hex(id(self.em))}>",
            ]
        )

    def __call__(self, x: FloatVector) -> FloatVector:
        return sum(
            (
                sample(x) 
                for sample in (self.pl, self.fe, self.ba, self.hg, self.em)
                if sample is not None
            ),
            start=zeros_like(x),
        )

    @property
    def parameters(self) -> FloatVector | None:
        lst: list[FloatVector] = []
        if self.pl is not None:
            lst.append(self.pl.parameters)
        if self.fe is not None:
            lst.append(self.fe.parameters)
        if self.ba is not None:
            lst.append(self.ba.parameters)
        if self.hg is not None:
            lst.append(self.hg.parameters)
        if self.em is not None:
            lst.append(self.em.parameters)
        if not lst:
            return None
        return concatenate(lst)

    @property
    def names(self) -> list[str] | None:
        lst: list[str] = []
        if self.pl is not None:
            lst.append(self.pl.name)
        if self.fe is not None:
            lst.extend(self.fe.names)
        if self.ba is not None:
            lst.append(self.ba.name)
        if self.hg is not None:
            lst.append(self.hg.name)
        if self.em is not None:
            lst.extend(self.em.names)
        return lst or None

    ### fromSpectrumSample

    @classmethod
    def _fromSpectrumSample(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        return {"x": x, "y": y, "dy": dy, "dx": dx, "weight": weight}

    @classmethod
    @validate_call
    def fromSpectrumSample(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector,
        weight: float = 1.0,
    ) -> Self:
        return Sample(**cls._fromSpectrumSample(x, y, dy, dx, weight=weight))

    ### fromDictOfSamples

    @classmethod
    def _fromDictOfSamples(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector,
        all_model_samples: dict[str, list[BaseSample]],
        weight: float = 1.0,
    ) -> dict[str, Any]:
        kwargs = {
            "x": x,
            "y": y,
            "dy": dy,
            "dx": dx,
            "pl": None,
            "fe": None,
            "ba": None,
            "hg": None,
            "em": None,
            "weight": weight,
        }
        for model_type, samples in all_model_samples.items():
            match model_type:
                case "pl":
                    if len(samples) == 1:
                        kwargs["pl"] = samples[0]
                    elif len(samples) > 1:
                        msg = "Allowing multiple power law models is not currently \
                            supported!"
                        raise NotImplementedError(msg)
                case "fe":
                    if len(samples) > 0:
                        kwargs["fe"] = IronSampleList.fromIronSamples(samples)
                case "ba":
                    if len(samples) == 1:
                        kwargs["ba"] = samples[0]
                    elif len(samples) > 1:
                        msg = "Allowing multiple Balmer models is not supported!"
                        raise NotImplementedError(msg)
                case "hg":
                    if len(samples) == 1:
                        kwargs["hg"] = samples[0]
                    elif len(samples) > 1:
                        msg = "Allowing multiple Host Galaxy models is not supported!"
                        raise NotImplementedError(msg)
                case "em":
                    if len(samples) > 0:
                        kwargs["em"] = GaussianSampleList.fromGaussianSamples(samples)

        return kwargs

    @classmethod
    @validate_call
    def fromDictOfSamples(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector,
        all_model_samples: dict[str, list] | None = None,
        weight: float = 1.0,
    ) -> Self:
        return cls._fromDictOfSamples(
            x, y, dy, dx, 
            all_model_samples or {}, 
            weight=weight,
        )

    ### fromModelSample

    @classmethod
    def _fromModelSample(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector,
        model: SequentialModel,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        all_model_samples: dict[str, list] = defaultdict(list)
        for submodel in model.submodels:
            key = submodel.model_type
            match key:
                case "pl":
                    sample = PowerLawSample.fromPowerLawModel(submodel)
                case "fe":
                    sample = IronSample.fromIronModel(submodel)
                case "ba":
                    sample = BalmerSample.fromBalmerModel(submodel)
                case "hg":
                    sample = HostGalaxySample.fromHostGalaxyModel(submodel)
                case "em":
                    sample = GaussianSample.fromGaussianModel(submodel)

            all_model_samples[key].append(sample)

        return cls._fromDictOfSamples(
            x, y, dy, dx, 
            all_model_samples, 
            weight=weight,
        )

    @classmethod
    @validate_call
    def fromModelSample(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector,
        model: SequentialModel,
        weight: float = 1.0,
    ) -> Self:
        return cls(**cls._fromModelSample(x, y, dy, dx, model, weight=weight))

    ### fromParamVector

    @classmethod
    def _fromParamVector(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector,
        param_vector: FloatVector,
        model: SequentialModel,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        all_model_samples: dict[str, list] = defaultdict(list)
        count: int = 0
        for submodel in model.submodels:
            key = submodel.model_type
            match key:
                case "pl":
                    sample_cls = PowerLawSample
                case "fe":
                    sample_cls = IronSample
                case "ba":
                    sample_cls = BalmerSample
                case "hg":
                    sample_cls = HostGalaxySample
                case "em":
                    sample_cls = GaussianSample

            n = submodel.parameters.size
            sample = sample_cls.fromParamVector(
                param_vector[count:count+n], 
                submodel,
            )
            all_model_samples[key].append(sample)
            count += n

        return cls._fromDictOfSamples(
            x, y, dy, dx, 
            all_model_samples, 
            weight=weight,
        )

    @classmethod
    @validate_call
    def fromParamVector(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        dx: FloatVector,
        param_vector: FloatVector,
        model: SequentialModel,
        weight: float = 1.0,
    ) -> Self:
        return cls(**cls._fromParamVector(
            x, y, dy, dx, 
            param_vector, model, 
            weight=weight,
        ))

    ### split

    def _split(
        self,
        method_name: str,
        target: Any,
        backup: Callable[[FloatVector], FloatVector] | None = None,
    ) -> tuple[Self, Self | None]:

        _cls = GaussianSampleList
        _cls_name = _cls.__name__
        if not hasattr(_cls, method_name):
            raise AttributeError(f"{_cls_name} has no attribute: {method_name}")
        if not callable(getattr(_cls, method_name)):
            raise TypeError(f"{_cls_name} has no callable method: {method_name}")

        if self.em is None:
            assert backup is not None
            target_sample = self.__class__(
                x=self.x,
                y=self.y - backup(self.x),
                dy=self.dy,
                dx=self.dx,
                pl=self.pl,
                fe=self.fe,
                ba=self.ba,
                hg=self.hg,
                em=None,
                weight=self.weight,
            )
            background_sample = None
        else:
            target_em, background_em = getattr(self.em, method_name)(target)
            if target_em is None:
                raise ValueError(
                    "Did not isolate any targetted emission component: "
                    f"{method_name=}, {target=}"
                )
            elif background_em is None:
                target_y = self.y.copy()
                background_y = zeros_like(self.x)
            else:
                target_y = self.y - background_em(self.x)
                background_y = self.y - target_em(self.x)

            target_sample = self.__class__(
                x=self.x, 
                y=target_y, 
                dy=self.dy, 
                dx=self.dx,
                pl=self.pl,
                fe=self.fe,
                ba=self.ba,
                hg=self.hg,
                em=target_em,
                weight=self.weight,
            )
            background_sample = self.__class__(
                x=self.x, 
                y=background_y, 
                dy=self.dy, 
                dx=self.dx,
                pl=self.pl,
                fe=self.fe,
                ba=self.ba,
                hg=self.hg, 
                em=background_em,
                weight=self.weight,
            )
        return target_sample, background_sample

    ### splitAtPattern

    def _splitAtPattern(
        self,
        pattern: StrPattern | tuple[StrPattern, ...] | Callable[[str], bool],
        backup: Callable[[FloatVector], FloatVector] | None = None,
    ) -> tuple[Self, Self | None]:
        return self._split("_splitAtPattern", pattern, backup=backup)

    @validate_call
    def splitAtPattern(
        self,
        pattern: StrPattern | tuple[StrPattern, ...] | Callable[[str], bool],
        backup: Callable[[FloatVector], FloatVector] | None = None,
    ) -> tuple[Self, Self | None]:
        return self._splitAtPattern(pattern, backup=backup)

    ### splitAtWave

    def _splitAtWave(
        self,
        target_wave: float | set[float],
        backup: Callable[[FloatVector], FloatVector] | None = None,
    ) -> tuple[Self, Self | None]:
        return self._split("_splitAtWave", target_wave, backup=backup)

    @validate_call
    def splitAtWave(
        self,
        target_wave: float | set[float],
        backup: Callable[[FloatVector], FloatVector] | None = None,
    ) -> tuple[Self, Self | None]:
        return self._splitAtWave(target_wave, backup=backup)

    ### splitAtLine

    def _splitAtLine(
        self,
        target_line: str | set[str],
        backup: Callable[[FloatVector], FloatVector] | None = None,
    ) -> tuple[Self, Self | None]:
        return self._split("_splitAtLine", target_line, backup=backup)

    @validate_call
    def splitAtLine(
        self,
        target_line: str | set[str],
        backup: Callable[[FloatVector], FloatVector] | None = None,
    ) -> tuple[Self, Self | None]:
        return self._splitAtLine(target_line, backup=backup)