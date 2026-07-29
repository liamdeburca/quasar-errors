from collections import defaultdict
from collections.abc import Callable
from typing import Self

from numpy import float64, zeros_like
from pydantic.dataclasses import dataclass
from quasar_typing.astropy import Model_
from quasar_typing.numpy import FloatVector
from quasar_utils.decorators import validate_call

from .balmer.balmer_sample import BalmerSample
from .continuum.powerlaw_sample import PowerLawSample
from .host.host_sample import HostGalaxySample
from .iron.iron_sample import IronSample
from .iron.iron_sample_list import IronSampleList
from .line.gaussian_sample import GaussianSample
from .line.gaussian_sample_list import GaussianSampleList


@dataclass(frozen=True, repr=False)
class Sample:
    x: FloatVector
    y: FloatVector
    dy: FloatVector

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
                f"  pl=<{'/' if self.pl is None else hex(id(self.pl))}>",
                f"  fe=<{'/' if self.fe is None else hex(id(self.fe))}>",
                f"  ba=<{'/' if self.ba is None else hex(id(self.ba))}>",
                f"  hg=<{'/' if self.hg is None else hex(id(self.hg))}>",
                f"  em=<{'/' if self.em is None else hex(id(self.em))}>",
            ]
        )

    def __call__(self, x: FloatVector) -> FloatVector:
        y = zeros_like(x, dtype=float64)
        for sample in (self.pl, self.fe, self.ba, self.hg, self.em):
            if sample is not None:
                y += sample(x)
        return y

    @classmethod
    @validate_call
    def fromSpectrumSample(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        weight: float = 1.0,
    ) -> Self:
        return Sample(x, y, dy, weight=weight)

    @classmethod
    @validate_call
    def fromModelSample(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        model: Model_,
        weight: float = 1.0,
    ) -> Self:

        pl_models = list(filter(lambda m: m.model_type == "pl", model))
        match len(pl_models):
            case 0:
                pl = None
            case 1:
                pl = PowerLawSample.fromPowerLawModel.__wrapped__(
                    PowerLawSample,
                    pl_models[0],
                )
            case _:
                msg = "Allowing multiple power law models is not currently \
                    supported!"
                raise NotImplementedError(msg)

        fe_models = list(filter(lambda m: m.model_type == "fe", model))
        match len(fe_models):
            case 0:
                fe = None
            case _:
                fe = IronSampleList.fromIronSubmodels.__wrapped__(
                    IronSampleList,
                    fe_models,
                )

        ba_models = list(filter(lambda m: m.model_type == "ba", model))
        match len(ba_models):
            case 0:
                ba = None
            case 1:
                ba = BalmerSample.fromBalmerModel.__wrapped__(
                    BalmerSample,
                    ba_models[0],
                )
            case _:
                msg = "Allowing multiple Balmer models is not supported!"
                raise NotImplementedError(msg)

        em_models = list(filter(lambda m: m.model_type == "em", model))
        match len(em_models):
            case 0:
                em = None
            case _:
                em = GaussianSampleList.fromGaussianSubmodels.__wrapped__(
                    GaussianSampleList,
                    em_models,
                )

        return Sample(x, y, dy, pl=pl, fe=fe, ba=ba, em=em, weight=weight)

    @classmethod
    @validate_call
    def fromParamVector(
        cls,
        x: FloatVector,
        y: FloatVector,
        dy: FloatVector,
        param_vector: FloatVector,
        model: Model_,
        weight: float = 1.0,
    ) -> Self:
        submodels = (model,) if model.n_submodels == 1 else model
        all_model_samples: dict[str, list] = defaultdict(list)

        count: int = 0
        for submodel in submodels:
            n = submodel.parameters.size
            match submodel.model_type, n:
                case "pl", _:
                    sample = PowerLawSample(
                        flux=param_vector[count],
                        alpha=param_vector[count + 1],
                        x0=submodel.x0,
                        name=submodel.name,
                        evaluate_func=submodel.evaluate_func,
                    )

                case "fe", _:
                    sample = IronSample(
                        flux=param_vector[count],
                        fwhm=param_vector[count + 1],
                        split=param_vector[count + 2],
                        left=param_vector[count + 3],
                        right=param_vector[count + 4],
                        name=submodel.name,
                        template=submodel.template,
                        evaluate_func=submodel.evaluate_func,
                        **submodel._interpolation_matrices,
                    )

                case "ba", _:
                    sample = BalmerSample(
                        flux=param_vector[count],
                        fwhm=param_vector[count + 1],
                        ratio=param_vector[count + 2],
                        name=submodel.name,
                        continuum_template=submodel.continuum_template,
                        series_template=submodel.series_template,
                        evaluate_func=submodel.evaluate_func,
                        **submodel._interpolation_matrices,
                    )
                case "hg", _:
                    sample = HostGalaxySample(
                        flux=param_vector[count],
                        fwhm=param_vector[count + 1],
                        name=submodel.name,
                        template=submodel.template,
                        evaluate_func=submodel.evaluate_func,
                        **submodel._interpolation_matrices,
                    )

                case "em", 3:
                    sample = GaussianSample(
                        strength=param_vector[count],
                        fwhm_v=param_vector[count + 1],
                        v_off=param_vector[count + 2],
                        wave=submodel.wave,
                        sigma_res=submodel.sigma_res,
                        n_sigmas=submodel.n_sigmas,
                        name=submodel.name,
                        pure_name=submodel.pure_name,
                        is_v_profile_copy=False,
                        n_profiles=1,
                        strength_scale=1.0,
                        evaluate_func=submodel.evaluate_func,
                    )
                case "em", _:
                    sample = GaussianSample(
                        strength=param_vector[count + 1 : count + n : 3],
                        fwhm_v=param_vector[count + 2 : count + n : 3],
                        v_off=param_vector[count + 3 : count + n : 3],
                        wave=submodel.wave,
                        sigma_res=submodel.sigma_res,
                        n_sigmas=submodel.n_sigmas,
                        name=submodel.name,
                        pure_name=submodel.pure_name,
                        is_v_profile_copy=True,
                        n_profiles=submodel._n_profiles,
                        strength_scale=param_vector[count],
                        # Leave out evaluate_func -> 'gaussian.evaluate_v'
                    )

            all_model_samples[submodel.model_type].append(sample)
            count += n

        pl_samples = all_model_samples["pl"]
        match len(pl_samples):
            case 0:
                pl = None
            case 1:
                pl = pl_samples[0]
            case _:
                msg = "Allowing multiple power law models is not currently \
                    supported!"
                raise NotImplementedError(msg)

        fe_samples = all_model_samples["fe"]
        match len(fe_samples):
            case 0:
                fe = None
            case _:
                fe = IronSampleList.fromIronSamples.__wrapped__(
                    IronSampleList,
                    fe_samples,
                )

        ba_samples = all_model_samples["ba"]
        match len(ba_samples):
            case 0:
                ba = None
            case 1:
                ba = ba_samples[0]
            case _:
                msg = "Allowing multiple Balmer models is not supported!"
                raise NotImplementedError(msg)

        em_samples = all_model_samples["em"]
        match len(em_samples):
            case 0:
                em = None
            case _:
                em = GaussianSampleList.fromGaussianSamples.__wrapped__(
                    GaussianSampleList,
                    em_samples,
                )

        return Sample(x, y, dy, pl=pl, fe=fe, ba=ba, em=em, weight=weight)

    @validate_call
    def splitAtWave(
        self,
        target_wave: float | set[float],
        backup: Callable[[FloatVector], FloatVector] | None = None,
    ) -> tuple[Self, Self | None]:

        if self.em is None:
            assert backup is not None

            selected_sample = Sample(
                self.x,
                self.y - backup(self.x),
                self.dy,
                pl=self.pl,
                fe=self.fe,
                ba=self.ba,
                weight=self.weight,
            )
            return selected_sample, None

        target_em, background_em = self.em.splitAtWave.__wrapped__(
            self.em,
            target_wave,
        )
        if target_em is None:
            target_y = zeros_like(self.x, dtype=float64)
            background_y = self.y.copy()
        elif background_em is None:
            target_y = self.y.copy()
            background_y = zeros_like(self.x, dtype=float64)
        else:
            target_y = self.y - background_em(self.x)
            background_y = self.y - target_em(self.x)

        target_sample = Sample(
            self.x,
            target_y,
            self.dy,
            pl=self.pl,
            fe=self.fe,
            ba=self.ba,
            em=target_em,
            weight=self.weight,
        )
        background_sample = Sample(
            self.x,
            background_y,
            self.dy,
            pl=self.pl,
            fe=self.fe,
            ba=self.ba,
            em=background_em,
            weight=self.weight,
        )

        return target_sample, background_sample

    @validate_call
    def splitAtLine(
        self,
        target_line: str | set[str],
        backup: Callable[[FloatVector], FloatVector] | None = None,
    ) -> tuple[Self, Self | None]:

        if self.em is None:
            assert backup is not None

            selected_sample = Sample(
                self.x,
                self.y - backup(self.x),
                self.dy,
                pl=self.pl,
                fe=self.fe,
                ba=self.ba,
                weight=self.weight,
            )
            return selected_sample, None

        target_em, background_em = self.em.splitAtLine.__wrapped__(
            self.em,
            target_line,
        )
        if target_em is None:
            target_y = zeros_like(self.x, dtype=float64)
            background_y = self.y.copy()
        elif background_em is None:
            target_y = self.y.copy()
            background_y = zeros_like(self.x, dtype=float64)
        else:
            target_y = self.y - background_em(self.x)
            background_y = self.y - target_em(self.x)

        target_sample = Sample(
            self.x,
            target_y,
            self.dy,
            pl=self.pl,
            fe=self.fe,
            ba=self.ba,
            em=target_em,
            weight=self.weight,
        )
        background_sample = Sample(
            self.x,
            background_y,
            self.dy,
            pl=self.pl,
            fe=self.fe,
            ba=self.ba,
            em=background_em,
            weight=self.weight,
        )

        return target_sample, background_sample
